import fitz  # PyMuPDF
import re

def extract_hindi_text(pdf_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    hindi_text = ""
    hindi_pattern = re.compile(r'[\u0900-\u097F]+')  # Unicode range for Hindi

    # Iterate through each page
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        
        # Find all Hindi text
        hindi_words = hindi_pattern.findall(text)
        if hindi_words:
            hindi_text += " ".join(hindi_words) + "\n"
    
    return hindi_text

if __name__ == "__main__":
    # Ask user for PDF input
    pdf_path = input("Enter the path to your PDF file: ")
    extracted_text = extract_hindi_text(pdf_path)
    
    # Output the extracted Hindi text
    if extracted_text:
        print("Extracted Hindi text:\n")
        print(extracted_text)
        #save in a text file
        with open("hindi_text.txt", "w", encoding="utf-8") as file:
            file.write(extracted_text)
        print("Hindi text saved in hindi text.txt file")
        
    else:
        print("No Hindi text found in the PDF.")
