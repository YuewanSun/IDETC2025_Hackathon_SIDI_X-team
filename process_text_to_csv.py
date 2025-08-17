import re
import csv
import os

def read_text_file(file_path):
    """Read the text file and return its content."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_into_sentences(text):
    """Split text into sentences while keeping rule references with their content."""
    # Replace newlines with spaces for easier processing
    text = re.sub(r'\s+', ' ', text)
    
    # Find sentence boundaries but avoid splitting rule references
    sentences = []
    pattern = r'(?<![A-Z]\.[0-9]|[0-9]\.[0-9])[\.\!\?]\s+(?=[A-Z0-9])'
    
    start = 0
    for match in re.finditer(pattern, text):
        end = match.end() - 1  # Include the punctuation
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
    
    # Add the last sentence
    if start < len(text):
        sentences.append(text[start:].strip())
    
    return sentences

def is_rule_reference(text):
    """Check if text starts with a rule reference like GR.1.4.2."""
    return bool(re.match(r'^[A-Z]+\.\d+(\.\d+)*', text.strip()))

def count_tokens(text):
    """Count the number of tokens (words) in text."""
    return len(text.split())

def chunk_sentences(sentences, max_tokens=250):
    """
    Chunk sentences into segments where:
    1. Each chunk starts with a rule reference (GR.1.3, etc.)
    2. Complete rules are kept together in a chunk
    3. Multiple rules can be in one chunk if they fit within token limit
    """
    chunks = []
    current_chunk = []
    current_token_count = 0
    current_rule = None
    
    for i, sentence in enumerate(sentences):
        # Check if this sentence starts a new rule
        is_rule_start = is_rule_reference(sentence)
        sentence_token_count = count_tokens(sentence)
        
        # Start a new chunk if:
        # 1. This is a rule reference AND
        # 2. We already have content in the current chunk AND
        # 3. Adding this would exceed our token limit
        if (is_rule_start and current_chunk and 
            (current_token_count + sentence_token_count > max_tokens)):
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_token_count = 0
        
        # If this is a rule start and current chunk is empty, or
        # if this continues the current rule, add it to current chunk
        if is_rule_start or not current_chunk or not is_rule_start:
            current_chunk.append(sentence)
            current_token_count += sentence_token_count
            if is_rule_start:
                current_rule = sentence
        
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def save_to_csv(chunks, output_file):
    """Save the chunks to a CSV file."""
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['chunk_id', 'text'])
        for i, chunk in enumerate(chunks):
            writer.writerow([i+1, chunk])

def main():
    # Input file path
    input_file = "../dataset/docs/rules_pdfplumber1.txt"
    # Output file path
    output_file = "../processed_dataset/rules_chunks.csv"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read the text file
    text = read_text_file(input_file)
    
    # Split into sentences
    sentences = split_into_sentences(text)
    
    # Chunk the sentences
    chunks = chunk_sentences(sentences, max_tokens=250)
    
    # Save to CSV
    save_to_csv(chunks, output_file)
    
    print(f"Processed {len(chunks)} chunks and saved to {output_file}")

if __name__ == "__main__":
    main()