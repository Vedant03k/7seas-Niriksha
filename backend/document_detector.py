"""
Niriksha Document Forgery Detector
Multi-layer analysis engine for detecting forged/AI-generated PDF documents.
Layer 1: Metadata Forensics   — Creation tool, timestamps, producer mismatch
Layer 2: Structural Analysis  — Hidden layers, embedded objects, font anomalies
Layer 3: Visual/Image Analysis — ELA on embedded images, compression artifacts
Layer 4: Text Consistency     — AI-tone indicators, date/number cross-checks
Layer 5: Digital Signature    — Presence and validity of PDF certificates
Final score = weighted ensemble of all layers.
"""

import os
import re
import hashlib
import tempfile
import subprocess
from datetime import datetime, timezone
from typing import Optional

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io


class DocumentForensicDetector:
    """Ensemble document forgery detector for PDFs."""

    W_METADATA   = 0.25
    W_STRUCTURE  = 0.20
    W_VISUAL     = 0.20
    W_TEXT       = 0.15
    W_SIGNATURE  = 0.20

    # Known AI/editing tool signatures in PDF metadata
    AI_TOOL_PATTERNS = [
        r"chatgpt", r"openai", r"gpt[-\s]?4",
        r"claude", r"anthropic", r"gemini", r"bard",
        r"jasper", r"copy\.?ai", r"writesonic",
        r"midjourney", r"dall[-\s]?e", r"stable.diffusion",
        r"canva", r"photoshop", r"illustrator", r"gimp",
        r"inkscape", r"affinity", r"corel",
        r"fpdf", r"reportlab", r"wkhtmltopdf", r"pdfkit",
        r"itext", r"pdfsharp", r"jspdf",
    ]

    LEGITIMATE_PRODUCERS = [
        r"microsoft", r"word", r"excel", r"powerpoint",
        r"libreoffice", r"openoffice", r"google.docs",
        r"acrobat", r"adobe", r"preview", r"cups",
        r"xerox", r"ricoh", r"hp\b", r"canon\b", r"epson",
        r"scansnap", r"naps2",
    ]

    def __init__(self):
        print("Document Forensic Detector initialized.")

    # ── Layer 1: Metadata Forensics ───────────────────────────────────────

    def _metadata_score(self, doc: fitz.Document, file_bytes: bytes):
        """Analyse PDF metadata for signs of forgery."""
        artifacts = []
        scores = []
        meta = doc.metadata or {}

        producer = (meta.get("producer") or "").strip()
        creator = (meta.get("creator") or "").strip()
        creation_date = meta.get("creationDate") or ""
        mod_date = meta.get("modDate") or ""
        title = (meta.get("title") or "").strip()
        author = (meta.get("author") or "").strip()

        combined_meta = f"{producer} {creator}".lower()

        # Check for AI tool signatures
        ai_tool_found = None
        for pattern in self.AI_TOOL_PATTERNS:
            if re.search(pattern, combined_meta, re.IGNORECASE):
                ai_tool_found = pattern
                break

        if ai_tool_found:
            scores.append(0.8)
            artifacts.append(f"AI/editing tool detected in metadata: producer='{producer}', creator='{creator}'")
        else:
            scores.append(0.0)

        # Check if producer matches a known legitimate source
        is_legitimate = any(
            re.search(p, combined_meta, re.IGNORECASE)
            for p in self.LEGITIMATE_PRODUCERS
        )
        if not is_legitimate and producer:
            scores.append(0.3)
            artifacts.append(f"Uncommon PDF producer: '{producer}' — not a standard office/scanner tool")
        elif not producer:
            scores.append(0.4)
            artifacts.append("Missing PDF producer metadata — legitimate documents usually have this")
        else:
            scores.append(0.0)

        # Parse and compare dates
        def parse_pdf_date(d):
            """Parse PDF date format D:YYYYMMDDHHmmSS"""
            match = re.search(r"D:(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?", d)
            if match:
                parts = [int(g) if g else 0 for g in match.groups()]
                try:
                    return datetime(parts[0], parts[1], parts[2],
                                    parts[3], parts[4], parts[5],
                                    tzinfo=timezone.utc)
                except ValueError:
                    return None
            return None

        created = parse_pdf_date(creation_date)
        modified = parse_pdf_date(mod_date)

        if created and modified:
            if modified < created:
                scores.append(0.7)
                artifacts.append(f"Modification date ({modified.date()}) is BEFORE creation date ({created.date()}) — impossible for legitimate documents")
            elif (modified - created).days > 3650:
                scores.append(0.4)
                artifacts.append(f"Document modified {(modified - created).days} days after creation — unusually long gap")
            else:
                scores.append(0.0)
        elif not creation_date and not mod_date:
            scores.append(0.3)
            artifacts.append("No creation or modification dates — metadata may have been stripped")
        else:
            scores.append(0.0)

        # Empty author/title on supposedly official docs
        if not author and not title:
            scores.append(0.2)
            artifacts.append("No author or title — official documents typically include these")
        else:
            scores.append(0.0)

        combined = float(0.5 * np.mean(scores) + 0.5 * max(scores)) if scores else 0.0
        return min(combined, 1.0), artifacts, {
            "producer": producer or "(none)",
            "creator": creator or "(none)",
            "author": author or "(none)",
            "title": title or "(none)",
            "creation_date": str(created.date()) if created else "(none)",
            "modification_date": str(modified.date()) if modified else "(none)",
            "ai_tool_detected": ai_tool_found,
            "file_hash_sha256": hashlib.sha256(file_bytes).hexdigest(),
            "file_size_kb": round(len(file_bytes) / 1024, 1),
        }

    # ── Layer 2: Structural Analysis ──────────────────────────────────────

    def _structure_score(self, doc: fitz.Document):
        """Check PDF structure for hidden layers, unusual objects, font anomalies."""
        artifacts = []
        scores = []
        page_count = len(doc)

        # --- Font analysis across pages ---
        all_fonts = set()
        font_per_page = []
        for page in doc:
            fonts = page.get_fonts(full=True)
            page_font_names = set()
            for f in fonts:
                fname = f[3] if len(f) > 3 else ""
                if fname:
                    page_font_names.add(fname)
                    all_fonts.add(fname)
            font_per_page.append(page_font_names)

        # Excessive font count (>10 different fonts is suspicious)
        if len(all_fonts) > 10:
            scores.append(0.5)
            artifacts.append(f"Excessive font variety ({len(all_fonts)} fonts) — suggests document was assembled from multiple sources")
        elif len(all_fonts) > 6:
            scores.append(0.3)
            artifacts.append(f"High font variety ({len(all_fonts)} fonts) — more than typical for a uniform document")
        else:
            scores.append(0.0)

        # Font inconsistency between pages
        if page_count > 1 and font_per_page:
            first_page_fonts = font_per_page[0]
            pages_with_different_fonts = sum(
                1 for fp in font_per_page[1:]
                if fp and fp != first_page_fonts
            )
            if pages_with_different_fonts > page_count * 0.5:
                scores.append(0.4)
                artifacts.append("Font sets vary significantly across pages — possible assembly from different sources")
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)

        # --- Embedded object analysis ---
        total_images = 0
        total_annotations = 0
        for page in doc:
            total_images += len(page.get_images(full=True))
            annots = page.annots()
            if annots:
                total_annotations += sum(1 for _ in annots)

        # Hidden annotations (form fields, redactions, etc.)
        if total_annotations > 0:
            scores.append(0.3)
            artifacts.append(f"{total_annotations} annotations/overlays found — document has editable layers")
        else:
            scores.append(0.0)

        # --- Image-only PDF (no real text, just embedded images) ---
        # Official documents saved as image-only PDFs are very suspicious
        total_text = ""
        for page in doc:
            total_text += page.get_text("text")
        has_text = len(total_text.strip()) > 30

        if not has_text and total_images > 0:
            scores.append(0.7)
            artifacts.append(
                f"Image-only PDF ({total_images} image(s), no extractable text) — "
                "legitimate documents have real text layers, not just images of text"
            )
        elif not has_text and total_images == 0 and len(all_fonts) == 0:
            scores.append(0.5)
            artifacts.append("PDF contains no text, fonts, or images — suspicious empty/generated document")
        else:
            scores.append(0.0)

        # --- Page size inconsistency ---
        if page_count > 1:
            sizes = []
            for page in doc:
                r = page.rect
                sizes.append((round(r.width, 1), round(r.height, 1)))
            unique_sizes = set(sizes)
            if len(unique_sizes) > 1:
                scores.append(0.5)
                artifacts.append(f"Inconsistent page sizes ({len(unique_sizes)} different sizes) — pages may come from different documents")
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)

        # --- JavaScript or embedded files (potential manipulation) ---
        xref_count = doc.xref_length()
        has_js = False
        has_embedded_files = False
        for i in range(1, min(xref_count, 500)):
            try:
                obj_str = doc.xref_object(i, compressed=False)
                if "/JavaScript" in obj_str or "/JS " in obj_str:
                    has_js = True
                if "/EmbeddedFile" in obj_str:
                    has_embedded_files = True
            except Exception:
                continue

        if has_js:
            scores.append(0.6)
            artifacts.append("JavaScript code embedded in PDF — unusual for standard documents")
        else:
            scores.append(0.0)

        if has_embedded_files:
            scores.append(0.3)
            artifacts.append("Embedded files found within PDF — could contain hidden content")
        else:
            scores.append(0.0)

        combined = float(0.5 * np.mean(scores) + 0.5 * max(scores)) if scores else 0.0
        return min(combined, 1.0), artifacts, {
            "page_count": page_count,
            "total_fonts": len(all_fonts),
            "font_names": sorted(list(all_fonts))[:20],
            "total_images": total_images,
            "total_annotations": total_annotations,
            "has_javascript": has_js,
            "has_embedded_files": has_embedded_files,
            "is_image_only": not has_text and total_images > 0,
        }

    # ── Layer 3: Visual/Image Analysis (ELA + AI detection) ─────────────

    def _visual_score(self, doc: fitz.Document, is_image_only: bool = False):
        """
        Extract embedded images and perform:
        - Error Level Analysis (ELA) for traditional editing
        - AI image detection via ImageDeepfakeDetector for AI-generated content
        """
        artifacts = []
        scores = []
        ela_details = []

        image_count = 0
        suspicious_images = 0
        image_bytes_list = []  # store raw PNG bytes for AI detection

        for page_idx, page in enumerate(doc):
            images = page.get_images(full=True)
            for img_idx, img_info in enumerate(images):
                xref = img_info[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:  # CMYK
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_data = pix.tobytes("png")
                    pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
                    image_count += 1
                    image_bytes_list.append(img_data)

                    # ELA: re-save at known quality, compare with original
                    ela_score = self._compute_ela(pil_img)
                    ela_details.append({
                        "page": page_idx + 1,
                        "image_index": img_idx,
                        "ela_score": round(ela_score, 3),
                        "dimensions": f"{pil_img.width}x{pil_img.height}",
                    })

                    if ela_score > 25:
                        suspicious_images += 1

                except Exception:
                    continue

        if image_count == 0:
            scores.append(0.0)
        else:
            suspicion_ratio = suspicious_images / image_count
            if suspicion_ratio > 0.5:
                scores.append(0.6)
                artifacts.append(f"{suspicious_images}/{image_count} embedded images show high ELA scores — possible editing/compositing")
            elif suspicion_ratio > 0.2:
                scores.append(0.3)
                artifacts.append(f"{suspicious_images}/{image_count} embedded images have elevated ELA — some regions may be modified")
            else:
                scores.append(0.0)

        # --- AI image detection on embedded images ---
        # Especially critical for image-only PDFs (AI-generated document images)
        ai_image_score = 0.0
        if image_bytes_list and is_image_only:
            try:
                from image_detector import ImageDeepfakeDetector
                img_detector = ImageDeepfakeDetector()
                # Analyze the largest/primary image
                primary_img = max(image_bytes_list, key=len)
                img_result = img_detector.analyze(primary_img)
                if img_result.get("status") != "error":
                    fake_prob = img_result.get("confidence", 0)
                    verdict = img_result.get("verdict", "REAL")
                    if verdict == "FAKE":
                        ai_image_score = min(fake_prob, 1.0)
                        scores.append(ai_image_score)
                        artifacts.append(
                            f"AI image detection flagged primary image as FAKE "
                            f"({fake_prob:.1%} confidence) — image appears AI-generated"
                        )
                        # Include specific artifacts from image detector
                        for art in img_result.get("artifacts_detected", [])[:3]:
                            artifacts.append(f"  → {art}")
                    elif verdict == "UNCERTAIN":
                        ai_image_score = fake_prob * 0.5
                        scores.append(ai_image_score)
                        artifacts.append(
                            f"AI image detection uncertain on primary image "
                            f"({fake_prob:.1%}) — could be AI-generated"
                        )
                    else:
                        scores.append(0.0)
            except Exception as e:
                # Image detector not available — continue without it
                scores.append(0.0)
        elif image_bytes_list and not is_image_only:
            scores.append(0.0)

        # Check for images with very different resolutions (copy-paste from different sources)
        if ela_details:
            widths = [int(d["dimensions"].split("x")[0]) for d in ela_details]
            heights = [int(d["dimensions"].split("x")[1]) for d in ela_details]
            if len(set(widths)) > 3 or len(set(heights)) > 3:
                scores.append(0.3)
                artifacts.append("Embedded images have widely varying resolutions — likely sourced from different documents")
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)

        combined = float(0.5 * np.mean(scores) + 0.5 * max(scores)) if scores else 0.0
        return min(combined, 1.0), artifacts, {
            "total_images_analyzed": image_count,
            "suspicious_images": suspicious_images,
            "ela_details": ela_details[:10],  # cap for response size
        }

    @staticmethod
    def _compute_ela(pil_img: Image.Image, quality: int = 90) -> float:
        """Error Level Analysis: re-save at known quality, measure difference."""
        buf = io.BytesIO()
        pil_img.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        resaved = Image.open(buf).convert("RGB")

        orig_arr = np.array(pil_img, dtype=np.float32)
        resaved_arr = np.array(resaved, dtype=np.float32)
        diff = np.abs(orig_arr - resaved_arr)
        return float(np.mean(diff))

    @staticmethod
    def _ocr_extract(doc: fitz.Document) -> str:
        """Extract text from image-only PDF pages using PyMuPDF's built-in OCR."""
        ocr_text = ""
        for page in doc:
            try:
                tp = page.get_textpage_ocr(flags=0, language="eng", dpi=150)
                ocr_text += page.get_text("text", textpage=tp) + "\n"
            except Exception:
                continue
        return ocr_text.strip()

    # ── Layer 4: Text Consistency Analysis ────────────────────────────────

    def _text_score(self, doc: fitz.Document, is_image_only: bool = False):
        """Analyse text for AI-generation indicators and internal inconsistencies.
        For image-only PDFs, attempt OCR first."""
        artifacts = []
        scores = []

        full_text = ""
        text_blocks_per_page = []
        for page in doc:
            text = page.get_text("text")
            full_text += text + "\n"
            text_blocks_per_page.append(text)

        ocr_text = ""
        # OCR fallback for image-only PDFs
        if len(full_text.strip()) < 50 and is_image_only:
            try:
                ocr_text = self._ocr_extract(doc)
                if len(ocr_text.strip()) > 50:
                    full_text = ocr_text
                    artifacts.append("Text extracted via OCR from image-only PDF")
            except Exception:
                pass

        if len(full_text.strip()) < 50:
            return 0.0, artifacts or ["Insufficient text for analysis"], {"text_length": len(full_text), "ocr_text": ocr_text}

        text_lower = full_text.lower()
        word_count = len(full_text.split())

        # --- AI phrasing indicators ---
        ai_phrases = [
            r"as an ai",
            r"i cannot provide",
            r"it['\u2019]s important to note",
            r"it is worth noting",
            r"in conclusion",
            r"as mentioned earlier",
            r"please note that",
            r"it should be noted",
            r"hereby certif(y|ies) that",  # not suspicious alone but weight context
            r"delve into",
            r"leverage\b",
            r"utilize\b",
            r"facilitate\b",
            r"comprehensive\b.*\banalysis\b",
        ]

        ai_phrase_count = 0
        for phrase in ai_phrases:
            matches = re.findall(phrase, text_lower)
            ai_phrase_count += len(matches)

        # Only flag if density is high (many AI phrases per 1000 words)
        ai_density = ai_phrase_count / max(word_count / 1000, 1)
        if ai_density > 5:
            scores.append(0.6)
            artifacts.append(f"High density of AI-typical phrases ({ai_phrase_count} occurrences, {ai_density:.1f}/1000 words)")
        elif ai_density > 2:
            scores.append(0.3)
            artifacts.append(f"Moderate AI-style phrasing detected ({ai_phrase_count} occurrences)")
        else:
            scores.append(0.0)

        # --- Date format consistency ---
        # Find all dates and check if format is consistent
        date_patterns = [
            (r"\d{1,2}/\d{1,2}/\d{2,4}", "MM/DD/YYYY"),
            (r"\d{1,2}-\d{1,2}-\d{2,4}", "MM-DD-YYYY"),
            (r"\d{4}-\d{2}-\d{2}", "YYYY-MM-DD"),
            (r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}", "Month DD, YYYY"),
        ]
        found_formats = set()
        for pattern, fmt_name in date_patterns:
            if re.search(pattern, text_lower):
                found_formats.add(fmt_name)

        if len(found_formats) > 2:
            scores.append(0.4)
            artifacts.append(f"Inconsistent date formats used ({', '.join(found_formats)}) — real documents typically use one format")
        else:
            scores.append(0.0)

        # --- Repetition analysis (AI tends to repeat sentence structures) ---
        sentences = re.split(r'[.!?]+', full_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        if len(sentences) > 10:
            # Check for repeated sentence starters
            starters = [' '.join(s.split()[:3]).lower() for s in sentences]
            from collections import Counter
            starter_counts = Counter(starters)
            repeated = sum(1 for c in starter_counts.values() if c >= 3)
            if repeated > 2:
                scores.append(0.4)
                artifacts.append(f"Repetitive sentence structures detected ({repeated} patterns repeated 3+ times)")
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)

        # --- Lorem ipsum / placeholder text ---
        if "lorem ipsum" in text_lower or "placeholder" in text_lower:
            scores.append(0.7)
            artifacts.append("Placeholder/Lorem Ipsum text found — incomplete document")
        else:
            scores.append(0.0)

        combined = float(0.5 * np.mean(scores) + 0.5 * max(scores)) if scores else 0.0
        return min(combined, 1.0), artifacts, {
            "word_count": word_count,
            "ai_phrase_count": ai_phrase_count,
            "ai_phrase_density": round(ai_density, 2),
            "date_formats_found": list(found_formats),
            "ocr_text": ocr_text,
        }

    # ── Layer 5: Digital Signature Verification ───────────────────────────

    def _signature_score(self, doc: fitz.Document, file_bytes: bytes, extra_text: str = ""):
        """
        Check for presence and validity of digital signatures/certificates.
        Legitimate official PDFs from banks, governments, law firms usually
        have digital signatures. Missing ones on 'official' docs are suspicious.
        """
        artifacts = []
        scores = []

        # Check for signature fields
        has_signature = False
        sig_count = 0
        for page in doc:
            try:
                widgets = page.widgets()
                if widgets:
                    for w in widgets:
                        if w.field_type_string == "Signature":
                            has_signature = True
                            sig_count += 1
            except Exception:
                continue

        # Also check xref for /Sig objects
        xref_count = doc.xref_length()
        for i in range(1, min(xref_count, 500)):
            try:
                obj_str = doc.xref_object(i, compressed=False)
                if "/Type /Sig" in obj_str or "/SubFilter" in obj_str:
                    has_signature = True
                    sig_count += 1
                    break
            except Exception:
                continue

        # Check if document text suggests it's official
        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + "\n"
        text_lower = full_text.lower()

        official_keywords = [
            "certificate", "certification", "hereby certif",
            "notarized", "notary", "official", "government",
            "bank statement", "invoice", "receipt",
            "contract", "agreement", "legal",
            "authorized signature", "seal",
        ]
        seems_official = sum(1 for kw in official_keywords if kw in text_lower)

        if seems_official >= 2 and not has_signature:
            scores.append(0.6)
            artifacts.append(f"Document appears official ({seems_official} formal keywords) but lacks digital signature/certificate")
        elif seems_official >= 1 and not has_signature:
            scores.append(0.3)
            artifacts.append("Document uses formal language but has no digital signature")
        elif has_signature:
            scores.append(0.0)
            # Don't add artifact — having a sig is good
        else:
            scores.append(0.1)

        # Check for certificate chain in the raw bytes
        has_cert = b"/ByteRange" in file_bytes or b"/Cert" in file_bytes
        if has_signature and not has_cert:
            scores.append(0.5)
            artifacts.append("Signature field exists but no embedded certificate chain — signature may be cosmetic/invalid")
        else:
            scores.append(0.0)

        combined = float(0.5 * np.mean(scores) + 0.5 * max(scores)) if scores else 0.0
        return min(combined, 1.0), artifacts, {
            "has_digital_signature": has_signature,
            "signature_count": sig_count,
            "has_certificate_chain": has_cert,
            "official_keyword_count": seems_official,
        }

    # ── Ensemble & Final Verdict ──────────────────────────────────────────

    def analyze_document(self, file_bytes: bytes, filename: str = "document.pdf"):
        """Run all analysis layers and produce ensemble verdict."""
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            return {
                "verdict": "ERROR",
                "confidence": 0,
                "media_type": "document",
                "artifacts_detected": [],
                "explanation": f"Failed to parse PDF: {str(e)}",
            }

        try:
            # Layer 1: Metadata
            meta_fake, meta_artifacts, meta_info = self._metadata_score(doc, file_bytes)

            # Layer 2: Structure
            struct_fake, struct_artifacts, struct_info = self._structure_score(doc)
            is_image_only = struct_info.get("is_image_only", False)

            # Layer 3: Visual/ELA + AI image detection
            visual_fake, visual_artifacts, visual_info = self._visual_score(doc, is_image_only=is_image_only)

            # Layer 4: Text (with OCR fallback for image-only PDFs)
            text_fake, text_artifacts, text_info = self._text_score(doc, is_image_only=is_image_only)
            ocr_text = text_info.get("ocr_text", "")

            # Layer 5: Signature
            sig_fake, sig_artifacts, sig_info = self._signature_score(doc, file_bytes, extra_text=ocr_text)

            # Weighted ensemble
            ensemble_fake = (
                self.W_METADATA  * meta_fake +
                self.W_STRUCTURE * struct_fake +
                self.W_VISUAL    * visual_fake +
                self.W_TEXT      * text_fake +
                self.W_SIGNATURE * sig_fake
            )

            # Multi-layer boost (similar to audio detector)
            layer_scores = [meta_fake, struct_fake, visual_fake, text_fake, sig_fake]
            flagging_layers = sum(1 for s in layer_scores if s > 0.2)
            strong_layers = sum(1 for s in layer_scores if s > 0.4)

            if flagging_layers >= 3 and strong_layers >= 1 and ensemble_fake < 0.5:
                avg_score = np.mean(layer_scores)
                max_score = max(layer_scores)
                boost = (avg_score + max_score) * 0.25
                ensemble_fake = min(ensemble_fake + boost, 0.90)

            ensemble_real = 1.0 - ensemble_fake
            verdict = "FAKE" if ensemble_fake > 0.5 else "REAL"
            confidence = max(ensemble_fake, ensemble_real)

            # Collect all artifacts
            all_artifacts = []
            all_artifacts.extend(meta_artifacts)
            all_artifacts.extend(struct_artifacts)
            all_artifacts.extend(visual_artifacts)
            all_artifacts.extend(text_artifacts)
            all_artifacts.extend(sig_artifacts)

            if verdict == "REAL" and not all_artifacts:
                all_artifacts.append("No forgery indicators detected across all analysis layers")

            if verdict == "FAKE":
                explanation = (
                    f"Multi-layer forensic analysis indicates this document is likely forged or "
                    f"AI-generated with {confidence:.1%} confidence. "
                    f"{flagging_layers} of 5 analysis layers flagged suspicious indicators."
                )
            else:
                explanation = (
                    f"Forensic analysis across metadata, structure, visual, text, and signature layers "
                    f"indicates this document appears authentic with {confidence:.1%} confidence."
                )

            return {
                "verdict": verdict,
                "confidence": round(confidence, 3),
                "media_type": "document",
                "filename": filename,
                "artifacts_detected": all_artifacts,
                "explanation": explanation,
                "sub_scores": {
                    "metadata_forensics": round(meta_fake, 4),
                    "structural_analysis": round(struct_fake, 4),
                    "visual_ela": round(visual_fake, 4),
                    "text_consistency": round(text_fake, 4),
                    "digital_signature": round(sig_fake, 4),
                },
                "metadata_info": meta_info,
                "structure_info": struct_info,
                "visual_info": visual_info,
                "text_info": text_info,
                "signature_info": sig_info,
            }

        except Exception as e:
            return {
                "verdict": "ERROR",
                "confidence": 0,
                "media_type": "document",
                "artifacts_detected": [],
                "explanation": f"Analysis failed: {str(e)}",
            }
        finally:
            doc.close()
