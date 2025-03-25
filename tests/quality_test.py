import pytest
from image_quality_checker import image_quality_score

def test_low_quality_image():
    small_image_bytes = b"\x00\x01\x02" * 1024  # Simulating a low-quality image
    assert image_quality_score(small_image_bytes) == True  # Expecting rejection

def test_valid_image():
    with open(r"C:\Users\wel\Pictures\Screenshots\Screenshot 2025-01-13 132259.png", "rb") as img_file:
        image_bytes = img_file.read()
    assert isinstance(image_quality_score(image_bytes), float)  # Should return float

if __name__ == "__main__":
    pytest.main()