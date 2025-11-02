"""
Flask API - Misinformation Detection with SEPARATE Endpoints
Fixed: Proper async handling, cleaner separation, two upload endpoints
EXTRA FIX: Now handles arrays of claims properly (because someone can't read docs)
"""

import asyncio
import os
import json
import traceback
from datetime import datetime
from functools import wraps
import logging
from concurrent.futures import ThreadPoolExecutor

try:
    from dotenv import load_dotenv
    load_dotenv("key.env")
except (ImportError, FileNotFoundError):
    pass

from flask import Flask, request, jsonify, copy_current_request_context
from werkzeug.utils import secure_filename
import google.generativeai as genai
from PIL import Image

# Import your modules
from scrappingAndFactcheck import initialize_fact_checker, cleanup_connections
from reverse_image_search import detect_image_misinformation

# (Authentication removed to simplify local testing)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Admin blueprint registration removed (auth disabled for testing)

UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

# ==================== FIXED: Better Async Route Decorator ====================

def async_route(f):
    """Run async functions in Flask routes without losing your mind"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        @copy_current_request_context
        def run_async():
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Run the async function while preserving Flask request/app context
                return loop.run_until_complete(f(*args, **kwargs))
            finally:
                loop.close()
        
        try:
            future = executor.submit(run_async)
            return future.result(timeout=120)  # 2 min timeout
        except Exception as e:
            logger.error(f"Async route explosion: {e}", exc_info=True)
            return jsonify({
                'error': f'Processing failed: {str(e)}',
                'status': 'epic_fail',
                'timestamp': datetime.now().isoformat()
            }), 500
    return wrapper


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


# ==================== SMART Text Extraction (For People Who Send Weird JSON) ====================

def extract_text_intelligently(data):
    """
    Extract actual text from whatever nonsense the user sends
    Handles: arrays, dicts, plain text, nested structures
    """
    logger.info(f"Extracting text from data type: {type(data)}")
    
    # If it's a list, process each item
    if isinstance(data, list):
        if len(data) == 0:
            return None, None
        
        # Take first item (or combine all)
        first_item = data[0]
        
        if isinstance(first_item, dict):
            # Extract text from first dict
            text = first_item.get('text') or first_item.get('claim') or first_item.get('content')
            claim_id = first_item.get('id', 'unknown')
            domain = first_item.get('domain', 'General')
            
            if text:
                logger.info(f"✅ Extracted text from array[0]: {text[:100]}...")
                return text, domain
        
        # If list of strings
        elif isinstance(first_item, str):
            return first_item, 'General'
    
    # If it's a dict
    elif isinstance(data, dict):
        text = data.get('text') or data.get('claim') or data.get('content') or data.get('claims')
        domain = data.get('domain', 'General')
        
        if text:
            logger.info(f"✅ Extracted text from dict: {text[:100]}...")
            return text, domain
    
    # If it's plain string
    elif isinstance(data, str):
        logger.info(f"✅ Plain string detected: {data[:100]}...")
        return data, 'General'
    
    # Fallback: stringify everything
    logger.warning("⚠️  Could not extract text properly, using fallback")
    return json.dumps(data), 'General'


# ==================== Gemini Text Extraction ====================

async def extract_text_with_gemini(image_path):
    """Extract text from image using Gemini Vision"""
    try:
        logger.info(f"Extracting text from: {image_path}")
        img = Image.open(image_path)
        if img.mode not in ('RGB', 'RGBA'):
            img = img.convert('RGB')
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = """Extract ALL text from this image. Include headlines, body text, captions.
        Return as JSON:
        {
            "headline": "main headline",
            "body": "main content",
            "metadata": "dates, sources",
            "additional": "other text"
        }"""
        
        response = await model.generate_content_async([prompt, img])
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        return ""


# ==================== Cross-Verification ====================

async def gemini_cross_verify(image_path, extracted_text, provided_text, reverse_search_results):
    """Cross-verify using Gemini AI"""
    try:
        logger.info("Running Gemini cross-verification...")
        img = Image.open(image_path)
        if img.mode not in ('RGB', 'RGBA'):
            img = img.convert('RGB')
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""Fact-check AI Analysis:

IMAGE TEXT: {extracted_text[:1000]}
PROVIDED CLAIM: {json.dumps(provided_text, indent=2)[:1000]}
REVERSE SEARCH: {json.dumps(reverse_search_results, indent=2)[:500]}

Return ONLY valid JSON:
{{
    "consistency_score": 0-100,
    "image_supports_claim": true/false,
    "inconsistencies_found": ["list"],
    "manipulation_indicators": ["list"],
    "context_verification": "text",
    "confidence_level": "high/medium/low",
    "detailed_analysis": "text"
}}"""

        response = await model.generate_content_async([prompt, img])
        
        try:
            result = json.loads(response.text.strip())
        except json.JSONDecodeError:
            result = {
                "consistency_score": 0,
                "image_supports_claim": False,
                "raw_analysis": response.text.strip(),
                "confidence_level": "low"
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Cross-verification died: {e}")
        return {"error": str(e), "confidence_level": "low"}


def extract_claims_from_data(text_data):
    """Extract claims from whatever format the user threw at us"""
    if isinstance(text_data, dict):
        if 'claims' in text_data:
            return text_data['claims'] if isinstance(text_data['claims'], list) else [text_data['claims']]
        elif 'text' in text_data:
            return [text_data['text']]
        else:
            return [str(text_data)]
    elif isinstance(text_data, list):
        return text_data
    else:
        return [str(text_data)]


def get_domain_from_text_data(text_data, default_domain='General'):
    """Derive domain from incoming text_data, with a sensible default."""
    try:
        if isinstance(text_data, dict):
            return text_data.get('domain') or default_domain
        if isinstance(text_data, list) and text_data:
            first = text_data[0]
            if isinstance(first, dict):
                return first.get('domain') or default_domain
        return default_domain
    except Exception:
        return default_domain


def generate_final_verdict(results):
    """Generate misinformation verdict - the moment of truth"""
    verdict = {
        'is_misinformation': False,
        'confidence_score': 0.0,
        'risk_level': 'UNKNOWN',
        'key_findings': [],
        'recommendations': []
    }
    
    try:
        image_confidence = results.get('image_analysis', {}).get('confidence_score', 0)
        cross_verify = results.get('cross_verification', {})
        consistency_score = cross_verify.get('consistency_score', 0)
        image_supports = cross_verify.get('image_supports_claim', True)
        
        # Weighted score calculation
        final_score = (image_confidence * 0.3 + consistency_score * 0.7)
        verdict['confidence_score'] = final_score
        
        # Misinformation detection
        if not image_supports or consistency_score < 50:
            verdict['is_misinformation'] = True
            verdict['risk_level'] = 'HIGH' if final_score > 70 else 'MEDIUM'
        elif final_score < 40:
            verdict['is_misinformation'] = True
            verdict['risk_level'] = 'LOW'
        else:
            verdict['risk_level'] = 'LOW'
        
        # Findings
        if cross_verify.get('inconsistencies_found'):
            verdict['key_findings'].extend(cross_verify['inconsistencies_found'])
        if cross_verify.get('manipulation_indicators'):
            verdict['key_findings'].extend(cross_verify['manipulation_indicators'])
        
        # Recommendations
        if verdict['is_misinformation']:
            verdict['recommendations'] = [
                "Verify with multiple trusted sources",
                "Check original source and date",
                "Look for fact-checks from reputable organizations",
                "Be cautious before sharing"
            ]
        else:
            verdict['recommendations'] = [
                "Content appears legitimate but always verify",
                "Cross-reference with other sources when possible"
            ]
        
    except Exception as e:
        logger.error(f"Verdict generation failed: {e}")
        verdict['error'] = str(e)
    
    return verdict


async def cross_verify_image_and_text(image_path, text_data, config):
    """Full cross-verification pipeline with enhanced debugging"""
    logger.info("="*70)
    logger.info("🔬 Starting FULL cross-verification pipeline...")
    logger.info(f"📥 Received text_data: {text_data}")
    logger.info(f"📥 Text_data type: {type(text_data)}")
    logger.info("="*70)
    
    results = {
        'image_analysis': {},
        'text_analysis': {},
        'cross_verification': {},
        'final_verdict': {},
        'timestamp': datetime.now().isoformat(),
        'debug_info': {
            'text_data_received': text_data,
            'text_data_type': str(type(text_data))
        }
    }
    
    try:
        # Step 1: Analyze provided text (extract claims first)
        logger.info("Step 1/5: Analyzing provided claims...")
        text_claims = extract_claims_from_data(text_data)
        results['text_analysis']['claims'] = text_claims
        domain = get_domain_from_text_data(text_data)
        results['text_analysis']['domain'] = domain
        logger.info(f"✅ Extracted claims: {text_claims}")

        # Concatenate claims for context
        claims_context = "; ".join([str(c) for c in text_claims])[:300]
        logger.info(f"📝 Claims context for search: {claims_context}")

        # NEW: Run structured text-only fact check using scrappingAndFactcheck
        # news_type will be auto-determined by scrappingAndFactcheck based on domain and text
        try:
            logger.info("Running structured text-only fact-check via scrappingAndFactcheck.initialize_fact_checker ...")
            logger.info(f"  - news_text: {claims_context[:200]}...")
            logger.info(f"  - misinformation_domain: {domain}")
            logger.info(f"  - news_type: Will be auto-determined by scrappingAndFactcheck module")
            
            fc = await initialize_fact_checker(
                news_text=claims_context or (text_claims[0] if text_claims else ""),
                misinformation_domain=domain
                # news_type is optional - will be auto-determined internally
            )
            
            # Store the complete fact-check result including all scraped content
            fc_dict = fc.to_dict()
            results['text_factcheck'] = fc_dict
            
            # Get the determined news_type from the fact-check result
            determined_news_type = fc.debug_data.get('news_type', 'Unknown')
            results['text_analysis']['news_type'] = determined_news_type
            logger.info(f"✅ Fact-check used news_type: {determined_news_type}")
            
            # Also store scraped content separately for easy access
            if fc.scraped_contents:
                results['scraped_contents'] = fc.scraped_contents
                results['scraped_content_count'] = len(fc.scraped_contents)
            if fc.trusted_urls:
                results['scraped_urls'] = fc.trusted_urls
                results['sources_used'] = fc.sources_used
            
            logger.info(f"✅ Text-only fact-check completed and merged")
            logger.info(f"   - Scraped {fc.scraped_content_count} contents from {len(fc.trusted_urls)} URLs")
            logger.info(f"   - Success: {fc.success}")
        except Exception as e:
            logger.error(f"❌ Text-only fact-check failed: {e}", exc_info=True)
            results['text_factcheck_error'] = str(e)

        # Step 2: Reverse image search (use claims context)
        logger.info("Step 2/5: Reverse image search with context...")
        image_results = await detect_image_misinformation(image_path, config, context_text=claims_context)
        results['image_analysis'] = image_results
        logger.info(f"✅ Image analysis completed. Found {len(image_results.get('sources_analyzed', []))} sources")
        
        # Step 3: Extract text from image
        logger.info("Step 3/5: Extracting text from image...")
        extracted_text = await extract_text_with_gemini(image_path)
        results['image_analysis']['extracted_text'] = extracted_text
        logger.info(f"✅ Extracted text: {extracted_text[:200]}...")
        
        # Step 4: Cross-verify with AI
        logger.info("Step 4/5: AI cross-verification...")
        verification = await gemini_cross_verify(
            image_path=image_path,
            extracted_text=extracted_text,
            provided_text=text_data,
            reverse_search_results=image_results.get('reverse_search_results', {})
        )
        results['cross_verification'] = verification
        logger.info(f"✅ Cross-verification completed. Consistency score: {verification.get('consistency_score', 'N/A')}")
        
        # Step 5: Gemini combined synthesis (IMAGE + SCRAPED CONTENT only; no direct user claim text)
        try:
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            model = genai.GenerativeModel('gemini-2.0-flash')

            # Build sources summary from image-based scraping
            scraped = image_results.get('scraped_content', []) or []
            sources_summary = "\n\n".join([
                f"Source {i+1}: {s.get('url','')}\nContent: {(s.get('content','') or '')[:400]}..."
                for i, s in enumerate(scraped)
            ]) or "No additional sources found"

            combined_prompt = f"""Analyze this image and provide a comprehensive verification assessment.

EVIDENCE TO CONSIDER:
- The image itself (provided with this prompt)
- Text visible in the image
- Information from online sources

TEXT IN IMAGE:
{(extracted_text or 'N/A')}

ONLINE SOURCES SUMMARY:
{sources_summary}

TASK: Provide a final, user-friendly determination (6-8 bullet points) that combines what you see in the image with information from online sources. 

Include:
- A clear verdict (True/False/Uncertain)
- Key evidence with references to sources
- Any contradictions found
- A brief recommendation

IMPORTANT: Write in clear, natural language that anyone can understand. Avoid technical terms like "OCR", "scraped", "extracted", or "reverse search". Focus on what the evidence shows, not how it was obtained.
"""

            # Send image along with the combined prompt
            loop = asyncio.get_event_loop()
            img_for_model = await loop.run_in_executor(None, Image.open, image_path)
            if img_for_model.mode not in ('RGB', 'RGBA'):
                img_for_model = img_for_model.convert('RGB')
            combined_resp = await model.generate_content_async([combined_prompt, img_for_model])
            results['combined_gemini_summary'] = (combined_resp.text or '').strip()
            # Surface sources for transparency
            results['sources_analyzed'] = image_results.get('sources_analyzed', [])
            results['sources_used'] = image_results.get('sources_used', [])
        except Exception as e:
            logger.warning(f"⚠️ Combined Gemini synthesis failed: {e}")

        # Step 6: Final verdict
        logger.info("Step 5/5: Generating final verdict...")
        final_verdict = generate_final_verdict(results)
        results['final_verdict'] = final_verdict
        logger.info(f"✅ Final verdict: {'MISINFORMATION' if final_verdict.get('is_misinformation') else 'LIKELY TRUE'}")
        
        logger.info("="*70)
        logger.info("✅ Cross-verification completed successfully")
        logger.info("="*70)
        return results
        
    except Exception as e:
        logger.error(f"❌ Cross-verification exploded: {e}", exc_info=True)
        results['error'] = str(e)
        results['error_traceback'] = traceback.format_exc()
        return results


# Also update the fixed endpoint code:

@app.route('/upload-image-text', methods=['POST'])
@async_route
async def upload_image_with_text():
    """
    ENDPOINT 1: Upload image + text (JSON) for FULL cross-verification
    Requires: Valid API key in X-API-Key header or api_key parameter
    """
    try:
        logger.info("="*70)
        logger.info("📸 ENDPOINT: /upload-image-text")
        logger.info("="*70)
        
        # Validate image
        if 'image' not in request.files:
            return jsonify({
                'error': 'Image is required for this endpoint',
                'hint': 'Use /upload-text-only for text-only verification',
                'status': 'failed'
            }), 400
        
        image_file = request.files['image']
        if not image_file or not allowed_file(image_file.filename):
            return jsonify({
                'error': 'Invalid image file. Allowed: png, jpg, jpeg, gif, bmp, webp',
                'status': 'failed'
            }), 400
        
        # Save image
        filename = secure_filename(image_file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{filename}")
        image_file.save(image_path)
        logger.info(f"✅ Image saved: {image_path}")
        
        # DEBUG: Log all request data
        logger.info("="*70)
        logger.info("🔍 DEBUG: Request inspection")
        logger.info(f"  Files keys: {list(request.files.keys())}")
        for key, f in request.files.items():
            logger.info(f"    - {key}: filename={f.filename}, mimetype={f.mimetype}, content_type={getattr(f, 'content_type', 'N/A')}")
        logger.info(f"  Form keys: {list(request.form.keys())}")
        for key in request.form.keys():
            logger.info(f"    - {key}: {request.form[key][:100] if len(request.form[key]) > 100 else request.form[key]}")
        logger.info(f"  Headers with 'json' or 'data': {[k for k in request.headers.keys() if 'json' in k.lower() or 'data' in k.lower()]}")
        logger.info("="*70)
        
        # Handle text data - TRY ALL METHODS
        text_data = None
        text_source = None
        
        # Method 1: JSON body (when Content-Type is application/json)
        if request.is_json:
            try:
                text_data = request.get_json(force=False)
                text_source = "json_body"
                logger.info(f"✅ Method 1: JSON body received via request.is_json")
            except Exception as e:
                logger.warning(f"⚠️ Method 1 failed: {e}")
        
        # Method 1b: Force JSON parsing even if Content-Type isn't set correctly
        if text_data is None:
            try:
                text_data = request.get_json(force=True, silent=True)
                if text_data is not None:
                    text_source = "json_body_forced"
                    logger.info(f"✅ Method 1b: JSON body parsed with force=True")
            except Exception as e:
                logger.debug(f"Method 1b (force JSON) not applicable: {e}")
        
        # Method 1c: Try parsing raw request data as JSON
        if text_data is None and request.get_data():
            try:
                request_data = request.get_data(as_text=True)
                if request_data and request_data.strip().startswith(('{', '[')):
                    text_data = json.loads(request_data)
                    text_source = "json_body_raw"
                    logger.info(f"✅ Method 1c: JSON parsed from raw request data")
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.debug(f"Method 1c (raw data parse) not JSON: {e}")
        
        # Method 2: JSON file in form-data - check ALL files except image
        if text_data is None and request.files:
            for key, file_obj in request.files.items():
                if key != 'image' and file_obj and file_obj.filename:
                    # Check by extension first, then try to parse as JSON
                    is_json_file = file_obj.filename.lower().endswith('.json') or file_obj.mimetype == 'application/json'
                    if is_json_file or key.lower() in ['json_file', 'json', 'claim', 'claims', 'text_data']:
                        try:
                            # Reset file pointer
                            file_obj.seek(0)
                            content = file_obj.read().decode('utf-8')
                            text_data = json.loads(content)
                            text_source = f"json_file_{key}"
                            logger.info(f"✅ Method 2: JSON file loaded from key '{key}' (filename: {file_obj.filename})")
                            break
                        except Exception as e:
                            logger.warning(f"⚠️ Method 2 failed for key '{key}': {e}")
                            # Try to read as plain text if JSON fails
                            try:
                                file_obj.seek(0)
                                content = file_obj.read().decode('utf-8')
                                if content.strip():
                                    text_data = {'text': content.strip()}
                                    text_source = f"text_file_{key}"
                                    logger.info(f"✅ Method 2: Plain text loaded from file '{key}'")
                                    break
                            except Exception:
                                pass
        
        # Method 3: Form fields
        if text_data is None:
            for form_key in ['text_data', 'claims', 'json', 'claim']:
                if form_key in request.form:
                    try:
                        raw = request.form[form_key]
                        text_data = json.loads(raw)
                        text_source = f"form_{form_key}"
                        logger.info(f"✅ Method 3: Parsed JSON from form field '{form_key}'")
                        break
                    except json.JSONDecodeError:
                        # Treat as plain text
                        text_data = {'text': request.form[form_key]}
                        text_source = f"form_{form_key}_plain"
                        logger.info(f"✅ Method 3: Plain text from form field '{form_key}'")
                        break
        
        # Method 4: X-JSON-Data header
        if text_data is None and 'X-JSON-Data' in request.headers:
            try:
                json_str = request.headers['X-JSON-Data']
                text_data = json.loads(json_str)
                text_source = "header_x_json_data"
                logger.info(f"✅ Method 4: JSON from X-JSON-Data header")
            except json.JSONDecodeError:
                text_data = {'text': request.headers['X-JSON-Data']}
                text_source = "header_x_json_data_plain"
                logger.info(f"✅ Method 4: Plain text from X-JSON-Data header")
        
        # CRITICAL LOGGING
        logger.info("="*70)
        logger.info(f"📋 TEXT DATA STATUS: {'✅ FOUND' if text_data else '❌ NOT FOUND'}")
        if text_data:
            logger.info(f"📋 Text source: {text_source}")
            logger.info(f"📋 Text data type: {type(text_data)}")
            logger.info(f"📋 Text data content: {json.dumps(text_data, indent=2)}")
        else:
            logger.warning("⚠️ NO TEXT DATA FOUND - will run image-only mode")
            logger.info("Available request data:")
            logger.info(f"  - Files: {list(request.files.keys())}")
            logger.info(f"  - Form: {list(request.form.keys())}")
            logger.info(f"  - Headers: {dict(request.headers)}")
        logger.info("="*70)
        
        # Configuration
        config = {
            "google_api_key": os.getenv('GOOGLE_API_KEY'),
            "google_cse_id": os.getenv('GOOGLE_CSE_ID'),
            "serpapi_key": None,
            "gemini_api_key": os.getenv('GEMINI_API_KEY'),
            "imgbb_api_key": os.getenv('IMGBB_API_KEY'),
        }
        
        # Perform analysis
        if text_data:
            logger.info("🔄 MODE: FULL CROSS-VERIFICATION (image + text)")
            logger.info(f"🔄 Calling cross_verify_image_and_text...")
            results = await cross_verify_image_and_text(image_path, text_data, config)
            analysis_type = "full_cross_verification"
        else:
            logger.info("🔄 MODE: IMAGE ONLY (no text data provided)")
            results = await detect_image_misinformation(image_path, config)
            analysis_type = "image_only"
        
        # Response
        response_data = {
            'endpoint': '/upload-image-text',
            'analysis_type': analysis_type,
            'text_data_provided': text_data is not None,
            'text_source': text_source,
            'results': results,
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        result_filename = f"image_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result_filepath = os.path.join(RESULTS_FOLDER, result_filename)
        
        with open(result_filepath, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, ensure_ascii=False, indent=2)
        
        response_data['results_file'] = result_filename
        
        logger.info("✅ Request completed successfully")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ Upload image+text failed: {e}", exc_info=True)
        return jsonify({
            'error': f'Processing failed: {str(e)}',
            'status': 'failed',
            'timestamp': datetime.now().isoformat()
        }), 500
@app.route('/upload-text-only', methods=['POST'])
@async_route
async def upload_text_only():
    """
    ENDPOINT 2: Upload text/JSON ONLY for fact-checking
    Requires: Valid API key in X-API-Key header or api_key parameter
    
    Accepts:
    - JSON body: {"text": "claim to check", "domain": "Health"}
    - JSON body (array): [{"id": "1", "text": "claim", "domain": "Health"}]
    - Form field: text_data (JSON string)
    - JSON file: json_file
    """
    try:
        logger.info("="*70)
        logger.info("📝 ENDPOINT: /upload-text-only")
        logger.info("="*70)
        
        raw_data = None
        
        # Handle JSON body
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request content type: {request.content_type}")
        logger.info(f"Request headers: {dict(request.headers)}")
        
        # Method 1: Try Flask's built-in JSON parsing (works if Content-Type is application/json)
        if request.is_json:
            try:
                raw_data = request.get_json(force=False)
                logger.info(f"✅ Method 1: JSON body received via request.is_json: {type(raw_data)}")
            except Exception as e:
                logger.warning(f"⚠️ Method 1 (is_json) failed: {e}")
        
        # Method 2: Force JSON parsing even if Content-Type isn't set correctly
        if raw_data is None:
            try:
                raw_data = request.get_json(force=True, silent=True)
                if raw_data is not None:
                    logger.info(f"✅ Method 2: JSON body parsed with force=True: {type(raw_data)}")
            except Exception as e:
                logger.warning(f"⚠️ Method 2 (force JSON) failed: {e}")
        
        # Method 3: Try parsing raw request data as JSON
        if raw_data is None and request.get_data():
            try:
                request_data = request.get_data(as_text=True)
                if request_data and request_data.strip().startswith(('{', '[')):
                    raw_data = json.loads(request_data)
                    logger.info(f"✅ Method 3: JSON parsed from raw request data: {type(raw_data)}")
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.debug(f"Method 3 (raw data parse) not JSON: {e}")
        
        # Method 4: Handle JSON file
        if raw_data is None and 'json_file' in request.files:
            json_file = request.files['json_file']
            if json_file and json_file.filename.endswith('.json'):
                try:
                    raw_data = json.load(json_file)
                    logger.info(f"✅ Method 4: JSON file loaded: {type(raw_data)}")
                except json.JSONDecodeError as e:
                    return jsonify({
                        'error': f'Invalid JSON file: {str(e)}',
                        'status': 'failed'
                    }), 400
        
        # Method 5: Handle form data
        if raw_data is None and 'text_data' in request.form:
            try:
                raw_data = json.loads(request.form['text_data'])
                logger.info(f"✅ Method 5: Form data parsed as JSON: {type(raw_data)}")
            except json.JSONDecodeError:
                raw_data = {'text': request.form['text_data']}
                logger.info(f"✅ Method 5: Plain text from form")
        
        if not raw_data:
            return jsonify({
                'error': 'No text data provided',
                'hint': 'Send JSON body, json_file, or text_data form field',
                'examples': {
                    'single_claim': '{"text": "Nepal PM wife died", "domain": "General"}',
                    'array_format': '[{"id": "1", "text": "Nepal PM wife died", "domain": "General"}]'
                },
                'status': 'failed'
            }), 400
        
        # SMART EXTRACTION
        text_content, domain = extract_text_intelligently(raw_data)
        
        if not text_content:
            return jsonify({
                'error': 'Could not extract text from provided data',
                'received_data_type': str(type(raw_data)),
                'status': 'failed'
            }), 400
        
        logger.info(f"📝 Text to check: {text_content[:200]}...")
        logger.info(f"🏷️  Domain: {domain}")
        logger.info(f"📰 news_type: Will be auto-determined by scrappingAndFactcheck module")
        
        # Perform fact-checking (news_type will be auto-determined internally)
        fact_check = await initialize_fact_checker(
            news_text=text_content,
            misinformation_domain=domain
            # news_type is optional - will be auto-determined internally
        )
        
        # Get the complete fact-check result
        fact_check_dict = fact_check.to_dict()
        
        # Get the determined news_type from the fact-check result
        determined_news_type = fact_check.debug_data.get('news_type', 'Unknown')
        logger.info(f"✅ Fact-check used news_type: {determined_news_type}")
        
        response_data = {
            'endpoint': '/upload-text-only',
            'analysis_type': 'text_only',
            'news_type': determined_news_type,
            'input_text': text_content[:500] + "..." if len(text_content) > 500 else text_content,
            'domain': domain,
            'results': fact_check_dict,
            'scraped_contents': fact_check.scraped_contents if fact_check.scraped_contents else [],
            'scraped_content_count': fact_check.scraped_content_count,
            'scraped_urls': fact_check.trusted_urls if fact_check.trusted_urls else [],
            'sources_used': fact_check.sources_used if fact_check.sources_used else [],
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        result_filename = f"text_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result_filepath = os.path.join(RESULTS_FOLDER, result_filename)
        
        with open(result_filepath, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, ensure_ascii=False, indent=2)
        
        response_data['results_file'] = result_filename
        
        logger.info("✅ Text-only fact-check completed")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Upload text-only failed: {e}", exc_info=True)
        return jsonify({
            'error': f'Processing failed: {str(e)}',
            'status': 'failed',
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check - is this thing even alive?"""
    env_vars = {
        'GEMINI_API_KEY': bool(os.getenv('GEMINI_API_KEY')),
        'GOOGLE_API_KEY': bool(os.getenv('GOOGLE_API_KEY')),
        'GOOGLE_CSE_ID': bool(os.getenv('GOOGLE_CSE_ID')),
        'SERPAPI_API_KEY': bool(os.getenv('SERPAPI_API_KEY')),
        'IMGBB_API_KEY': bool(os.getenv('IMGBB_API_KEY')),
    }
    
    all_required = env_vars['GEMINI_API_KEY'] and env_vars['GOOGLE_API_KEY'] and env_vars['GOOGLE_CSE_ID']
    
    return jsonify({
        'status': 'healthy' if all_required else 'barely_alive',
        'message': 'Misinformation Detection API - Now with SMART text extraction!',
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            '/health': 'GET - Health check',
            '/upload-image-text': 'POST - Image + Text cross-verification',
            '/upload-text-only': 'POST - Text-only fact-checking (handles arrays now!)',
        },
        'environment': env_vars,
        'warnings': [] if all_required else ['Missing required API keys - this will explode soon']
    })


if __name__ == "__main__":
    # NOTE: admin initialization removed to avoid auto-generating API keys on restart
    print("="*70)
    print("🚀 Misinformation Detection API (auth disabled for local testing)")
    print("="*70)
    print("\n📡 Endpoints:")
    print("   POST /upload-image-text  - Image + Text verification")
    print("   POST /upload-text-only   - Text-only fact-checking")
    print("   GET  /health             - Health check")
    print("\n💡 Supported JSON formats:")
    print("   • Single: {\"text\": \"claim\", \"domain\": \"Health\"}")
    print("   • Array: [{\"id\": \"1\", \"text\": \"claim\"}]")
    print("   • Plain text")
    print("\n⚙️  Required Environment Variables:")
    print("   ✓ GEMINI_API_KEY" if os.getenv('GEMINI_API_KEY') else "   ✗ GEMINI_API_KEY (MISSING)")
    print("   ✓ GOOGLE_API_KEY" if os.getenv('GOOGLE_API_KEY') else "   ✗ GOOGLE_API_KEY (MISSING)")
    print("   ✓ GOOGLE_CSE_ID" if os.getenv('GOOGLE_CSE_ID') else "   ✗ GOOGLE_CSE_ID (MISSING)")
    print("   ✓ SERPAPI_API_KEY" if os.getenv('SERPAPI_API_KEY') else "   ⚠  SERPAPI_API_KEY (Optional)")
    print("   ✓ IMGBB_API_KEY" if os.getenv('IMGBB_API_KEY') else "   ⚠  IMGBB_API_KEY (Optional)")
    print("\n" + "="*70)
    
    app.run(debug=True, host='0.0.0.0', port=8000, threaded=True)