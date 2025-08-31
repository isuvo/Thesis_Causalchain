"""
Utility functions for hashing and content deduplication.
"""
import hashlib

def md5_hash(content):
    """
    Calculates the MD5 hash of the given content.
    """
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def main():
    # Example usage
    text = "Hello, world!"
    print(f"The MD5 hash of '{text}' is: {md5_hash(text)}")

if __name__ == "__main__":
    main()
