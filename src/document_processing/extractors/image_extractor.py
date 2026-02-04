# cockatoo_v1/src/document_processing/extractors/image_extractor.py

"""
Image file extractor with OCR support using Tesseract.
"""

import os
import logging
from typing import Dict, Any, List, Union, Optional
from pathlib import Path
import tempfile

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

# Try to import image processing libraries
try:
    from PIL import Image, ImageOps, ImageEnhance
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("PIL/Pillow not installed. Image processing will be limited.")

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    logger.warning("pytesseract not installed. OCR will not be available.")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("OpenCV not installed. Advanced image processing will be limited.")


class ImageExtractor(BaseExtractor):
    """
    Image file extractor with OCR capabilities.
    """
    
    def __init__(self, use_ocr: bool = True, ocr_languages: List[str] = None):
        """
        Initialize image extractor.
        
        Args:
            use_ocr: Whether to use OCR for text extraction
            ocr_languages: List of OCR languages (e.g., ['eng', 'ind'])
        """
        super().__init__()
        self.use_ocr = use_ocr and HAS_TESSERACT
        self.ocr_languages = ocr_languages or ['eng']
        
        if use_ocr and not HAS_TESSERACT:
            self.logger.warning(
                "pytesseract is not installed. OCR will be disabled. "
                "Install with: pip install pytesseract"
            )
            self.logger.info("Also ensure Tesseract OCR engine is installed on your system")
        
        if not HAS_PIL:
            self.logger.warning(
                "PIL/Pillow is not installed. Install with: pip install Pillow"
            )
    
    def get_supported_formats(self) -> List[str]:
        """
        Return list of supported image formats.
        
        Returns:
            List of supported file extensions
        """
        return ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.ico', '.svg']
    
    def extract(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract text and metadata from image file.
        
        Args:
            file_path: Path to image file
        
        Returns:
            Dictionary containing:
                - text: Extracted text via OCR
                - metadata: Image metadata
                - image_info: Image properties
                - ocr_info: OCR processing details
                - preprocessed: Preprocessing steps applied
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        logger.info(f"Extracting image: {file_path}")
        
        result = {
            "text": "",
            "metadata": self.get_basic_metadata(file_path),
            "image_info": {},
            "ocr_info": {},
            "preprocessed": {},
            "has_text": False,
            "extraction_method": "ocr" if self.use_ocr else "metadata_only",
        }
        
        # Extract basic image information
        result = self._extract_image_info(file_path, result)
        
        # Perform OCR if enabled and supported
        if self.use_ocr and HAS_PIL:
            try:
                result = self._perform_ocr(file_path, result)
            except Exception as e:
                logger.error(f"OCR extraction failed: {e}")
                result["ocr_info"]["error"] = str(e)
        
        # Add language detection if text was extracted
        if result["text"]:
            result["metadata"]["language"] = self.detect_language(result["text"])
            result["metadata"]["summary"] = self.extract_summary(result["text"])
            result["has_text"] = True
        
        return result
    
    def _extract_image_info(self, file_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract basic image information.
        
        Args:
            file_path: Path to image file
            result: Result dictionary to update
            
        Returns:
            Updated result dictionary
        """
        image_info = {
            "file_size": file_path.stat().st_size,
            "format": file_path.suffix.lower(),
            "is_valid": False,
        }
        
        if HAS_PIL:
            try:
                with Image.open(file_path) as img:
                    # Get basic image properties
                    image_info.update({
                        "is_valid": True,
                        "width": img.width,
                        "height": img.height,
                        "mode": img.mode,
                        "format": img.format,
                        "animated": getattr(img, "is_animated", False),
                        "frames": getattr(img, "n_frames", 1),
                    })
                    
                    # Extract EXIF data if available
                    exif_data = {}
                    try:
                        exif = img._getexif()
                        if exif:
                            # Common EXIF tags
                            exif_tags = {
                                271: "manufacturer",
                                272: "model",
                                274: "orientation",
                                306: "datetime",
                                36867: "datetime_original",
                                36868: "datetime_digitized",
                                33434: "exposure_time",
                                33437: "f_number",
                                34855: "iso_speed",
                                37378: "focal_length",
                                37383: "metering_mode",
                                37385: "flash",
                                37500: "maker_note",
                                42016: "image_unique_id",
                            }
                            
                            for tag_id, value in exif.items():
                                tag_name = exif_tags.get(tag_id, f"tag_{tag_id}")
                                exif_data[tag_name] = str(value)
                    except:
                        pass
                    
                    image_info["exif"] = exif_data
                    
                    # Calculate image hash for identification
                    try:
                        image_hash = self._calculate_image_hash(img)
                        image_info["hash"] = image_hash
                    except:
                        pass
                    
            except Exception as e:
                image_info["error"] = str(e)
                logger.error(f"Failed to read image info: {e}")
        
        elif HAS_CV2:
            try:
                img = cv2.imread(str(file_path))
                if img is not None:
                    image_info.update({
                        "is_valid": True,
                        "width": img.shape[1],
                        "height": img.shape[0],
                        "channels": img.shape[2] if len(img.shape) > 2 else 1,
                        "dtype": str(img.dtype),
                    })
            except Exception as e:
                image_info["error"] = str(e)
        
        result["image_info"] = image_info
        
        return result
    
    def _perform_ocr(self, file_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform OCR on image file.
        
        Args:
            file_path: Path to image file
            result: Result dictionary to update
            
        Returns:
            Updated result dictionary
        """
        ocr_info = {
            "languages": self.ocr_languages,
            "confidence": 0.0,
            "boxes": [],
            "preprocessing_steps": [],
            "success": False,
        }
        
        try:
            # Preprocess image for better OCR
            preprocessed_image, preprocessing_info = self._preprocess_image(file_path)
            ocr_info["preprocessing_steps"] = preprocessing_info
            result["preprocessed"] = preprocessing_info
            
            # Configure OCR
            config = self._get_ocr_config()
            
            # Perform OCR
            start_time = os.times().elapsed
            
            # Extract text with confidence
            data = pytesseract.image_to_data(
                preprocessed_image,
                lang='+'.join(self.ocr_languages),
                output_type=pytesseract.Output.DICT,
                config=config
            )
            
            processing_time = os.times().elapsed - start_time
            
            # Extract text and confidence
            text_blocks = []
            confidences = []
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = float(data['conf'][i]) if data['conf'][i] != '-1' else 0
                
                if text:
                    text_blocks.append(text)
                    confidences.append(conf)
                    
                    # Store bounding box information
                    if int(data['conf'][i]) > 0:  # Only store valid detections
                        box_info = {
                            "text": text,
                            "confidence": conf,
                            "x": data['left'][i],
                            "y": data['top'][i],
                            "width": data['width'][i],
                            "height": data['height'][i],
                            "block_num": data['block_num'][i],
                            "line_num": data['line_num'][i],
                            "word_num": data['word_num'][i],
                        }
                        ocr_info["boxes"].append(box_info)
            
            # Combine text
            extracted_text = ' '.join(text_blocks)
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            ocr_info.update({
                "success": True,
                "confidence": avg_confidence,
                "text_blocks_count": len(text_blocks),
                "processing_time_seconds": processing_time,
                "detected_languages": self._detect_ocr_language(extracted_text),
            })
            
            result.update({
                "text": self.clean_text(extracted_text),
                "ocr_info": ocr_info,
            })
            
        except Exception as e:
            ocr_info["error"] = str(e)
            ocr_info["success"] = False
            result["ocr_info"] = ocr_info
        
        return result
    
    def _preprocess_image(self, file_path: Path):
        """
        Preprocess image for better OCR results.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Tuple of (preprocessed_image, preprocessing_info)
        """
        preprocessing_info = []
        
        try:
            # Open image
            image = Image.open(file_path)
            original_mode = image.mode
            
            # Convert to grayscale if not already
            if image.mode != 'L':
                image = image.convert('L')
                preprocessing_info.append("converted_to_grayscale")
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)  # Increase contrast
            preprocessing_info.append("enhanced_contrast")
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            preprocessing_info.append("enhanced_sharpness")
            
            # Resize if too small (minimum 300px on shortest side)
            min_size = 300
            if min(image.width, image.height) < min_size:
                ratio = min_size / min(image.width, image.height)
                new_width = int(image.width * ratio)
                new_height = int(image.height * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                preprocessing_info.append(f"resized_{image.width}x{image.height}")
            
            # Apply thresholding for binary image (optional)
            # image = image.point(lambda x: 0 if x < 128 else 255, '1')
            # preprocessing_info.append("applied_threshold")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                image.save(tmp.name, 'PNG')
                temp_path = tmp.name
            
            return temp_path, preprocessing_info
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            # Return original file if preprocessing fails
            return str(file_path), ["preprocessing_failed"]
    
    def _get_ocr_config(self) -> str:
        """
        Get OCR configuration string for Tesseract.
        
        Returns:
            Tesseract configuration string
        """
        config_parts = []
        
        # Page segmentation mode
        config_parts.append('--psm 3')  # Fully automatic page segmentation, but no OSD
        
        # OCR engine mode
        config_parts.append('--oem 3')  # Default OCR engine mode
        
        # Other optimizations
        config_parts.append('-c preserve_interword_spaces=1')
        config_parts.append('-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:-_()[]{}@#$%&*+=\\/|"\' ')
        
        return ' '.join(config_parts)
    
    def _calculate_image_hash(self, image: Image.Image) -> str:
        """
        Calculate perceptual hash of image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Image hash string
        """
        import hashlib
        
        # Convert to grayscale and resize to 8x8
        img = image.convert('L').resize((8, 8), Image.Resampling.LANCZOS)
        
        # Get pixel values
        pixels = list(img.getdata())
        
        # Calculate average
        avg = sum(pixels) / len(pixels)
        
        # Create hash
        hash_str = ''
        for pixel in pixels:
            if pixel > avg:
                hash_str += '1'
            else:
                hash_str += '0'
        
        # Convert binary string to hex
        hash_hex = hex(int(hash_str, 2))[2:].zfill(16)
        
        return hash_hex
    
    def _detect_ocr_language(self, text: str) -> List[str]:
        """
        Detect languages in OCR text.
        
        Args:
            text: Extracted text
            
        Returns:
            List of detected language codes
        """
        if not text or len(text) < 10:
            return ["unknown"]
        
        # Simple detection based on character sets
        # This is simplified - consider using langdetect for production
        
        # Check for common language patterns
        languages = []
        
        # English
        english_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        english_count = sum(1 for c in text if c in english_chars)
        
        # Indonesian
        indonesian_special = set('áéíóúÁÉÍÓÚ')
        indonesian_count = sum(1 for c in text if c in indonesian_special)
        
        if english_count > indonesian_count and english_count > len(text) * 0.5:
            languages.append("en")
        
        if indonesian_count > 0 or 'yang' in text.lower() or 'dan' in text.lower():
            languages.append("id")
        
        if not languages:
            languages.append("unknown")
        
        return languages
    
    def extract_without_ocr(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract only image metadata without OCR.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Image metadata
        """
        result = {
            "metadata": self.get_basic_metadata(file_path),
            "image_info": {},
            "text": "",
            "has_text": False,
        }
        
        result = self._extract_image_info(file_path, result)
        
        return result
    
    def detect_text_regions(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Detect text regions in image without full OCR.
        
        Args:
            file_path: Path to image file
            
        Returns:
            List of text regions with bounding boxes
        """
        if not (HAS_TESSERACT and HAS_PIL):
            return []
        
        regions = []
        
        try:
            # Use Tesseract to detect text regions
            image = Image.open(file_path)
            
            # Get text detection data
            data = pytesseract.image_to_boxes(
                image,
                lang='+'.join(self.ocr_languages),
                output_type=pytesseract.Output.DICT
            )
            
            # Group characters into words
            current_word = ""
            current_box = None
            
            for i in range(len(data['char'])):
                char = data['char'][i]
                left = data['left'][i]
                right = data['right'][i]
                top = data['top'][i]
                bottom = data['bottom'][i]
                
                if char == ' ':
                    if current_word and current_box:
                        regions.append({
                            "text": current_word,
                            "x1": current_box[0],
                            "y1": current_box[1],
                            "x2": current_box[2],
                            "y2": current_box[3],
                            "width": current_box[2] - current_box[0],
                            "height": current_box[3] - current_box[1],
                        })
                    current_word = ""
                    current_box = None
                else:
                    current_word += char
                    if current_box is None:
                        current_box = [left, top, right, bottom]
                    else:
                        current_box[0] = min(current_box[0], left)
                        current_box[1] = min(current_box[1], top)
                        current_box[2] = max(current_box[2], right)
                        current_box[3] = max(current_box[3], bottom)
            
            # Add last word if exists
            if current_word and current_box:
                regions.append({
                    "text": current_word,
                    "x1": current_box[0],
                    "y1": current_box[1],
                    "x2": current_box[2],
                    "y2": current_box[3],
                    "width": current_box[2] - current_box[0],
                    "height": current_box[3] - current_box[1],
                })
                
        except Exception as e:
            logger.error(f"Failed to detect text regions: {e}")
        
        return regions