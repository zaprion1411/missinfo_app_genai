"""
Fixed: reverse_image_search.py
All blocking calls replaced with async versions
"""

import os
import logging
import asyncio
import aiohttp
import base64
from bs4 import BeautifulSoup
from PIL import Image, UnidentifiedImageError
import google.generativeai as genai


logger = logging.getLogger(__name__)

# ===== Helper: safe image opener =====
def safe_open_image(image_path):
    """Try to open an image file with PIL and return Image object or None.
    This avoids unhandled UnidentifiedImageError bubbles and centralizes logging.
    """
    try:
        img = Image.open(image_path)
        return img
    except UnidentifiedImageError:
        logger.error(f"Invalid image file or unsupported format: {image_path}")
        return None
    except Exception as e:
        logger.error(f"Error opening image {image_path}: {e}")
        return None

# ============= ASYNC Image Upload to ImgBB =============

async def upload_image_to_imgbb(image_path, imgbb_api_key=None):
    """Upload image to ImgBB to get public URL (ASYNC)"""
    if not imgbb_api_key:
        logger.error("IMGBB_API_KEY not provided! Get free key from https://api.imgbb.com/")
        return None
    
    logger.info(f"Uploading image to ImgBB: {image_path}")
    
    try:
        with open(image_path, 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.imgbb.com/1/upload",
                data={'key': imgbb_api_key, 'image': img_base64},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get('success'):
                        image_url = result['data']['url']
                        logger.info(f"✅ Image uploaded: {image_url}")
                        return image_url
                
                error_text = await response.text()
                logger.error(f"ImgBB upload failed: {error_text}")
                return None
        
    except Exception as e:
        logger.error(f"ImgBB upload error: {e}")
        return None


# NOTE: SerpAPI integration removed to simplify local testing and avoid
# dependency on the serpapi package/api. We rely on Google CSE + OCR fallback
# and ImgBB upload (if IMGBB_API_KEY is provided) for local images.


# ============= Fallback: Google CSE with Image Context =============

async def google_search_with_image_context(image_path, google_api_key, google_cse_id, context_text=None):
    """Fallback: Extract text from image, then search (ASYNC).
    If context_text is provided (e.g., user claim), blend it with OCR text for a more precise query.
    """
    logger.info("Using fallback: Google search with image context...")
    
    try:
        try:
            import pytesseract
        except ImportError:
            logger.error("pytesseract not installed")
            return {
                "error": "pytesseract not installed: pip install pytesseract",
                "success": False
            }
        
        if not google_api_key or not google_cse_id:
            logger.error(f"Missing Google credentials - API key: {bool(google_api_key)}, CSE ID: {bool(google_cse_id)}")
            return {
                "error": "Google API key or CSE ID not provided",
                "success": False
            }
        
        # Extract text
        loop = asyncio.get_event_loop()
        img = await loop.run_in_executor(None, safe_open_image, image_path)
        if img is None:
            logger.error(f"Fallback search failed: cannot open image file '{image_path}'")
            return {
                "error": f"Cannot open image file: {image_path}",
                "success": False
            }
        extracted_text = await loop.run_in_executor(None, pytesseract.image_to_string, img)
        extracted_text = extracted_text.strip()
        
        if not extracted_text:
            logger.warning("No text found in image using OCR")
            return {
                "error": "No text found in image for context search",
                "success": False
            }
        
        logger.info(f"Extracted text: {extracted_text[:100]}...")

        # Build a precise query: clean OCR noise, quote key phrase, restrict to fact-check domains
        def build_precise_query(text: str) -> str:
            try:
                import re
                # Normalize whitespace and remove obvious noise characters
                normalized = re.sub(r"[\n\r\t]+", " ", text)
                normalized = re.sub(r"[^\w\s\-:'\,\.?]", " ", normalized)
                normalized = re.sub(r"\s+", " ", normalized).strip()

                # If we have context_text (user-provided), blend it with priority
                blended = normalized
                if context_text:
                    ctx = re.sub(r"[\n\r\t]+", " ", str(context_text))
                    ctx = re.sub(r"[^\w\s\-:'\,\.?]", " ", ctx)
                    ctx = re.sub(r"\s+", " ", ctx).strip()
                    # Put context first; keep length reasonable
                    blended = (ctx + " | " + normalized)[:280]

                # Take a reasonable slice for key phrase
                key_phrase = blended[:140]
                # Prefer taking up to the first punctuation as a headline-like phrase
                m = re.search(r"(.{20,140}?[\.!?])", blended)
                if m:
                    key_phrase = m.group(1)

                # Quote the key phrase to force exact match behavior
                quoted = f'"{key_phrase}"'

                # Reputable fact-check domains
                domains = [
                    "site:snopes.com",
                    "site:factcheck.org",
                    "site:politifact.com",
                    "site:altnews.in",
                    "site:boomlive.in",
                    "site:afp.com",
                    "site:reuters.com/fact-check",
                    "site:apnews.com/fact-check",
                    "site:bbc.co.uk/news/bbcverify",
                    "site:bbc.com/news/verify",
                    "site:boomfactcheck.com"
                ]

                domain_filter = " OR ".join(domains)
                # Bias towards fact-check intent
                intent = "fact check debunk verify"
                return f"{quoted} {intent} ({domain_filter})"
            except Exception:
                # Safe fallback if anything goes wrong
                return f"{text[:120]} fact check misinformation"

        search_query = build_precise_query(extracted_text)
        
        params = {
            "key": google_api_key,
            "cx": google_cse_id,
            "q": search_query,
            "num": 10
        }
        
        async with aiohttp.ClientSession() as session:
            async def run_cse(query: str, post_filter: bool = True):
                local_params = dict(params)
                local_params["q"] = query
                async with session.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params=local_params,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(f"Google CSE API failed with status {resp.status}: {text}")
                        return []
                    data = await resp.json()
                    items = data.get("items", [])
                    results_list = []
                    for item in items:
                        link = item.get("link") or ""
                        title = item.get("title")
                        snippet = item.get("snippet")
                        if post_filter:
                            preferred = any(d in link for d in [
                                "snopes.com", "factcheck.org", "politifact.com", "altnews.in",
                                "boomlive.in", "reuters.com/fact-check", "apnews.com/fact-check",
                                "bbc.co.uk/news/bbcverify", "bbc.com/news/verify", "afp.com"
                            ])
                            if not preferred and link.rstrip('/').count('/') < 3:
                                continue
                        results_list.append({
                            "title": title,
                            "link": link,
                            "snippet": snippet
                        })
                    return results_list

            # First pass: precise fact-check–biased query
            search_results = await run_cse(search_query, post_filter=True)

            # If nothing found, relax constraints and retry with broader query
            if not search_results:
                logger.info("No matches on fact-check domains; retrying with broader query...")
                # Build broader query: remove domain filter, keep quoted phrase and intent
                try:
                    quoted_phrase = search_query.split('"')[1]
                except Exception:
                    quoted_phrase = extracted_text[:120]
                broad_query = f'"{quoted_phrase}" verify debunk'
                search_results = await run_cse(broad_query, post_filter=False)

            return {
                "method": "text_extraction_fallback",
                "extracted_text": extracted_text,
                "search_results": search_results,
                "total_results": len(search_results),
                "success": True
            }
        
    except Exception as e:
        logger.error(f"Fallback search failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e), "success": False}


async def google_cse_text_search(query_text, google_api_key, google_cse_id):
    """Perform a plain Google CSE query based on provided text to collect candidate sources."""
    try:
        if not google_api_key or not google_cse_id:
            return {"success": False, "error": "Missing Google credentials"}

        # Light normalization
        import re
        qt = re.sub(r"\s+", " ", (query_text or "").strip())
        if not qt:
            return {"success": False, "error": "Empty query"}

        params = {
            "key": google_api_key,
            "cx": google_cse_id,
            "q": qt,
            "num": 10
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    return {"success": False, "error": f"CSE status {resp.status}"}
                data = await resp.json()
                items = data.get("items", [])
                search_results = [
                    {
                        "title": it.get("title"),
                        "link": it.get("link"),
                        "snippet": it.get("snippet")
                    }
                    for it in items if it.get("link")
                ]
                return {"success": True, "search_results": search_results}
    except Exception as e:
        logger.error(f"Text CSE search failed: {e}")
        return {"success": False, "error": str(e)}

# ============= Image-only Analysis with Gemini =============

async def analyze_image_only_with_gemini(image_path, gemini_api_key):
    """Analyze only the image to assess misinformation indicators using Gemini Vision."""
    try:
        if not gemini_api_key:
            return {
                "fact_check_assessment": "AI analysis skipped: missing GEMINI_API_KEY",
                "confidence_score": 0.0,
                "success": False
            }

        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')

        prompt = (
            "You are verifying an image for potential misinformation. "
            "Without external links, analyze ONLY the visual content: text present, logos, watermarks, signs of editing,"
            " and whether the content likely needs further verification. Return 4 concise lines:"
            "\n1) Verdict: True/False/Uncertain"
            "\n2) Rationale: ..."
            "\n3) RedFlags: ..."
            "\n4) Confidence: 0-100"
        )

        loop = asyncio.get_event_loop()
        img = await loop.run_in_executor(None, safe_open_image, image_path)
        if img is None:
            logger.error(f"Invalid image for Gemini image-only analysis: {image_path}")
            # Fall back to text-only prompt (no image)
            response = await model.generate_content_async(prompt)
        else:
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            response = await model.generate_content_async([prompt, img])
        assessment = response.text.strip() if getattr(response, 'text', None) else ""

        # Heuristic confidence extraction
        conf = 0.5
        try:
            import re
            m = re.search(r"Confidence:\s*(\d{1,3})", assessment, re.IGNORECASE)
            if m:
                conf = max(0, min(100, int(m.group(1)))) / 100.0
        except Exception:
            pass

        return {
            "fact_check_assessment": assessment or "Image-only AI analysis completed.",
            "confidence_score": conf,
            "success": True
        }
    except Exception as e:
        logger.error(f"Gemini image-only analysis failed: {e}")
        return {
            "fact_check_assessment": f"AI analysis failed: {e}",
            "confidence_score": 0.0,
            "success": False
        }


# ============= Scrape Content =============

async def scrape_url_content(session, url):
    """Scrape content from URL"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9"
    }
    
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20), headers=headers, allow_redirects=True) as response:
            response.raise_for_status()
            html_content = await response.text()
            soup = BeautifulSoup(html_content, 'html.parser')
            
            paragraphs = soup.find_all('p')
            text_content = ' '.join([p.get_text() for p in paragraphs])
            
            if not text_content:
                text_content = soup.get_text()
            
            date_meta = soup.find('meta', property='article:published_time')
            publish_date = date_meta['content'] if date_meta else "Unknown"
            
            return {
                "url": url,
                "content": text_content.strip()[:2000],
                "publish_date": publish_date,
                "status_code": response.status,
                "success": True
            }
            
    except Exception as e:
        logger.warning(f"Failed to scrape {url}: {e}")
        return {
            "url": url,
            "content": "",
            "error": str(e),
            "status_code": getattr(e, 'status', None) if hasattr(e, 'status') else None,
            "success": False
        }


async def scrape_multiple_urls(urls):
    """Scrape multiple URLs in parallel"""
    logger.info(f"Scraping content from {len(urls)} URLs...")
    
    async with aiohttp.ClientSession() as session:
        tasks = [scrape_url_content(session, url) for url in urls[:8]]
        results = await asyncio.gather(*tasks)
        
    successful = [r for r in results if r.get('success')]
    logger.info(f"Successfully scraped {len(successful)} pages")
    
    return successful


# ============= MAIN DETECTION PIPELINE =============

async def detect_image_misinformation(image_path, config, context_text=None):
    """Complete pipeline: Reverse search → Scrape → Analyze.
    Optionally uses context_text (user-provided claim) to guide fallback search.
    """
    logger.info("=" * 70)
    logger.info("🚀 STARTING IMAGE MISINFORMATION DETECTION")
    logger.info("=" * 70)
    
    results = {
        "image_path": image_path,
        "reverse_search_results": {},
        "scraped_content": [],
        "fact_check_assessment": "",
        "misinformation_indicators": [],
        "sources_analyzed": [],
        "confidence_score": 0.0,
        "text_extracted": "",
        "provided_context": (context_text or "")
    }
    
    # Step 1: Reverse search
    logger.info("\n📸 Step 1: Performing reverse image search...")
    
    # SerpAPI intentionally disabled per requirements; always use CSE+Gemini path
    serpapi_key = None
    google_key = config.get("google_api_key")
    google_cse = config.get("google_cse_id")
    
    reverse_results = None
    
    # Skip SerpAPI entirely
    
    if not reverse_results or not reverse_results.get("success"):
        logger.warning("⚠️  SerpAPI failed; attempting Google CSE fallback instead of early return...")
        if google_key and google_cse:
            reverse_results = await google_search_with_image_context(
                image_path, google_key, google_cse, context_text=context_text
            )
            results["reverse_search_results"] = reverse_results
            results["search_method"] = "google_cse_fallback"
            results["text_extracted"] = reverse_results.get("extracted_text", "")
        else:
            # Proceed without external sources; we'll still run Gemini analysis later
            results["reverse_search_results"] = {"method": "skipped", "success": False}
            results["search_method"] = "none_available"
    
    if not reverse_results or not reverse_results.get("success"):
        logger.error("❌ All search methods failed")
        results["fact_check_assessment"] = f"Could not perform reverse search: {reverse_results.get('error', 'Unknown error')}"
        return results
    
    # Step 2: Extract URLs
    logger.info("\n🔗 Step 2: Extracting sources...")
    
    urls_to_scrape = []
    
    if "text_results" in reverse_results:
        urls_to_scrape = [r["link"] for r in reverse_results["text_results"][:8] if r.get("link")]
    elif "search_results" in reverse_results:
        urls_to_scrape = [r["link"] for r in reverse_results["search_results"][:8] if r.get("link")]
    
    if not urls_to_scrape:
        logger.warning("⚠️  No sources found from image search")
        # NEW: If we have context_text, try a text-only CSE to fetch sources
        if context_text:
            logger.info("🔄 Falling back to text-only CSE using provided context to gather sources...")
            text_cse = await google_cse_text_search(context_text, google_key, google_cse)
            if text_cse.get("success"):
                urls_to_scrape = [r["link"] for r in text_cse.get("search_results", []) if r.get("link")][:8]
                logger.info(f"✅ Text CSE provided {len(urls_to_scrape)} sources")
            else:
                logger.warning(f"⚠️ Text CSE failed: {text_cse.get('error')}")
        if not urls_to_scrape:
            logger.warning("⚠️  Still no sources; continuing to AI analysis with OCR only")
    
    logger.info(f"✅ Found {len(urls_to_scrape)} sources to analyze")
    results["sources_analyzed"] = urls_to_scrape
    
    # Step 3: Scrape
    logger.info("\n📄 Step 3: Scraping source content...")
    scraped_data = await scrape_multiple_urls(urls_to_scrape) if urls_to_scrape else []
    results["scraped_content"] = scraped_data
    # Expose explicit per-source status summary
    results["sources_used"] = [
        {
            "url": s.get("url"),
            "success": s.get("success", False),
            "status_code": s.get("status_code"),
            "content_length": len(s.get("content", "")) if s.get("content") else 0,
            "publish_date": s.get("publish_date")
        }
        for s in scraped_data
    ]
    
    # Step 4: Analyze with Gemini
    logger.info("\n🤖 Step 4: AI analysis...")

    gemini_key = config.get("gemini_api_key")
    if gemini_key:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-2.0-flash')

        sources_summary = "\n\n".join([
            f"Source {i+1}: {s['url']}\nContent: {s['content'][:300]}..."
            for i, s in enumerate(scraped_data)
        ])

        # Ensure we have extracted text (from OCR fallback or CSE)
        extracted_text = results.get("text_extracted") or ""
        if not extracted_text:
            try:
                import pytesseract
                loop = asyncio.get_event_loop()
                img_ocr = await loop.run_in_executor(None, safe_open_image, image_path)
                if img_ocr is not None:
                    extracted_text = await loop.run_in_executor(None, pytesseract.image_to_string, img_ocr)
                    extracted_text = (extracted_text or "").strip()
                    results["text_extracted"] = extracted_text
                else:
                    extracted_text = ""
                    results["text_extracted"] = ""
            except Exception:
                extracted_text = ""

        # Do NOT send user's claim to Gemini; use only OCR + scraped content
        logger.info("📝 Context text NOT sent to Gemini; using OCR + scraped content only")

        prompt = f"""Analyze this image for potential misinformation. Use the following information:

TEXT FOUND IN THE IMAGE:
{(extracted_text or 'N/A')}

RELEVANT ONLINE SOURCES:
{(sources_summary or 'No additional sources found')}

TASK: Based on what you see in the image and the information from online sources above, provide a clear assessment.

Your response should include:
1. Verdict: True/False/Uncertain
2. Rationale: Explain what the image shows and how it compares to information from online sources
3. Red flags: Note any signs of editing, watermarks, inconsistencies, or out-of-context use
4. Confidence: Rate your confidence level (0-100)

Write in clear, natural language. Avoid technical terms like "OCR" or "scraped" in your response."""

        try:
            # Send image alongside the prompt
            loop = asyncio.get_event_loop()
            img_for_model = await loop.run_in_executor(None, safe_open_image, image_path)
            if img_for_model is not None:
                if img_for_model.mode not in ('RGB', 'RGBA'):
                    img_for_model = img_for_model.convert('RGB')
                response = await model.generate_content_async([prompt, img_for_model])
            else:
                # If image is invalid, send text-only prompt
                logger.warning(f"Image invalid for Gemini model; sending text-only prompt for {image_path}")
                response = await model.generate_content_async(prompt)

            assessment = response.text.strip() if getattr(response, 'text', None) else ''
            results["fact_check_assessment"] = assessment

            if "high confidence" in assessment.lower():
                results["confidence_score"] = 0.85
            elif "moderate" in assessment.lower():
                results["confidence_score"] = 0.65
            else:
                results["confidence_score"] = 0.4

        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            results["fact_check_assessment"] = f"AI analysis failed: {e}"

        # Optional: Produce sources summary if we have sources
        if gemini_key and scraped_data:
            try:
                scraped_prompt = (
                    "Summarize the following online sources in 4-6 clear bullet points. Focus on verifiable facts, dates, and locations. "
                    "Write in natural, user-friendly language. Do not mention technical processes or how information was gathered.\n\n" + sources_summary
                )
                loop = asyncio.get_event_loop()
                scraped_resp = await model.generate_content_async(scraped_prompt)
                results["scraped_summary"] = (scraped_resp.text or "").strip()
            except Exception:
                # Heuristic fallback if Gemini errors
                try:
                    top = scraped_data[0]
                    results["scraped_summary"] = f"Top source: {top.get('url','')} — {top.get('content','')[:200]}..."
                except Exception:
                    results["scraped_summary"] = ""

        # Final synthesis summary combining image + OCR + sources (no user claim text)
        if gemini_key:
            try:
                final_prompt = f"""Create a final, user-friendly summary (5-7 bullet points) that combines evidence from:
1) What you see in the image
2) Text visible in the image
3) Information from online sources

Include: a clear verdict (True/False/Uncertain), key evidence with references, any contradictions found, and a brief recommendation.

TEXT FROM IMAGE:
{(extracted_text or 'N/A')}

SOURCES SUMMARY:
{results.get('scraped_summary','N/A')}

IMPORTANT: Write in natural, accessible language. Do not use technical terms like "OCR", "scraped", or "extracted". Focus on what the evidence tells us, not how it was gathered.
"""
                final_resp = await model.generate_content_async(final_prompt)
                results["final_summary"] = (final_resp.text or "").strip()
            except Exception:
                results["final_summary"] = results.get("fact_check_assessment", "")
    
    # Step 5: Detect indicators
    logger.info("\n🚨 Step 5: Detecting misinformation indicators...")
    
    indicators = []
    fact_check_domains = [
        "snopes.com", "factcheck.org", "politifact.com",
        "altnews.in", "boomlive.in", "afp.com",
        "reuters.com/fact-check", "apnews.com/fact-check",
        "bbc.co.uk/news/bbcverify", "bbc.com/news/verify"
    ]
    
    search_items = reverse_results.get("text_results", []) or reverse_results.get("search_results", [])
    
    for item in search_items:
        link = item.get("link", "")
        title = item.get("title", "").lower()
        snippet = item.get("snippet", "").lower()
        
        if any(domain in link for domain in fact_check_domains):
            indicators.append(f"⚠️  Found on fact-check site: {link}")
        
        if any(word in title or word in snippet for word in ["false", "fake", "debunk", "hoax"]):
            indicators.append(f"⚠️  Potential debunking: {item.get('title')}")
    
    results["misinformation_indicators"] = indicators
    results["sources_found"] = urls_to_scrape
    
    # Further education resources
    results["further_education"] = [
        "How to verify images: reverse search on Google/Bing/Yandex, and check dates",
        "Search exact quoted phrases from OCR text to find original context",
        "Consult trusted fact-checkers: Snopes, PolitiFact, FactCheck.org, Reuters/AP Fact Check",
        "Beware of cropped screenshots, watermarks, mismatched fonts or lighting",
        "Cross-check names/places in the image with reputable news sources"
    ]
    
    logger.info("=" * 70)
    logger.info("✅ ANALYSIS COMPLETE")
    logger.info("=" * 70)
    
    return results


if __name__ == "__main__":
    test_config = {
        "serpapi_key": os.getenv('SERPAPI_API_KEY'),
        "google_api_key": os.getenv('GOOGLE_API_KEY'),
        "google_cse_id": os.getenv('GOOGLE_CSE_ID'),
        "gemini_api_key": os.getenv('GEMINI_API_KEY'),
        "imgbb_api_key": os.getenv('IMGBB_API_KEY')
    }
    
    test_image = "test_image.jpg"
    
    if os.path.exists(test_image):
        result = asyncio.run(detect_image_misinformation(test_image, test_config))
        print("\n📊 RESULTS:")
        print(f"Method: {result.get('search_method')}")
        print(f"Sources: {len(result.get('sources_analyzed', []))}")
        print(f"Confidence: {result.get('confidence_score')}")
        print(f"\nAssessment:\n{result.get('fact_check_assessment')}")
    else:
        print(f"Create {test_image} to test!")