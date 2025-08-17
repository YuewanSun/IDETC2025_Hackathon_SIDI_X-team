import torch
import json
from transformers import MllamaForConditionalGeneration, AutoProcessor

def extract_keywords_and_rules(chunk_text):
    """
    Extract keywords and rule numbers from a given chunk of text.
    
    Args:
        chunk_text (str): A chunk of text from the rules document
        
    Returns:
        dict: Dictionary containing extracted keywords and rule numbers
    """
    model_id = "C:\Software\Model\llama3\Llama-3.2-11B-Vision-Instruct"
    
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Create prompt to extract keywords and rule numbers with JSON schema
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": f"""
I need to extract important information from Formula SAE rule text. Here's an example:

Example text:
VE.2.4 Electrical Equipment (EV Only) The following items must accompany the vehicle at all times: Two pairs of High Voltage insulating gloves  A multimeter Formula SAE Rules 2023 SAE International VE.2.5 Camera Mounts VE.2.5.1 The mounts for video/photographic cameras must be of a safe and secure design. VE.2.5.2 All camera installations must be approved at Technical Inspection. VE.2.5.3 Helmet mounted cameras and helmet camera mounts are prohibited. VE.2.5.4 The body of a camera or recording unit that weighs more than 0.25 kg must be secured at a minimum of two points on different sides of the camera body. VE.2.5.5 If a tether is used to restrain the camera, the tether length must be limited to prevent contact with the driver. VE.3 DRIVER EQUIPMENT VE.3.1 General VE.3.1.1 Any Driver Equipment: a. Must be in good condition with no tears, rips, open seams, areas of significant wear, abrasions or stains which might compromise performance. b. Must fit properly c. May be inspected at any time VE.3.1.2 Flame Resistant Material For this section some, but not all, of the approved materials are: Carbon X, Indura, Nomex, Polybenzimidazole (common name PBI) and Proban. VE.3.1.3 Synthetic Material â€" Prohibited Shirts, socks or other undergarments (not to be confused with flame resistant underwear) made from nylon or any other synthetic material which could melt when exposed to high heat are prohibited.

Example output:
{{
  "rule_numbers": ["VE.2.4", "VE.2.5", "VE.2.5.1", "VE.2.5.2", "VE.2.5.3", "VE.2.5.4", "VE.2.5.5", "VE.3", "VE.3.1", "VE.3.1.1", "VE.3.1.2", "VE.3.1.3"],
  "keywords": ["Electrical Equipment", "High Voltage insulating gloves", "Camera Mounts", "Helmet", "Technical Inspection", "Driver Equipment", "Flame Resistant Material", "Synthetic Material"]
}}

Now, analyze this new text:

{chunk_text}

Extract the following information:
1. All rule numbers (like GR.1.4.2) in this text
2. Important technical keywords or key concepts from this text

Return ONLY a valid JSON object with the following schema:
{{
    "rule_numbers": ["GR.1.2.3", "GR.4.5.6", ...],
    "keywords": ["keyword1", "keyword2", "keyword3", ...]
}}

The response should be valid JSON with no additional text before or after.
"""}
        ]}
    ]
    
    
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=input_text, return_tensors="pt").to(model.device)
    
    # Generate output
    output = model.generate(**inputs, max_new_tokens=500)
    response = processor.decode(output[0], skip_special_tokens=True)
    
    # Find JSON in the response - models sometimes add text before/after the JSON
    try:
        # Look for JSON pattern between curly braces
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            result = json.loads(json_str)
            # Ensure required keys exist
            if "rule_numbers" not in result:
                result["rule_numbers"] = []
            if "keywords" not in result:
                result["keywords"] = []
            return result
    except json.JSONDecodeError:
        pass
    
    # Fallback to old parsing method if JSON parsing fails
    result = {"rule_numbers": [], "keywords": []}
    for line in response.split('\n'):
        if line.startswith('"rule_numbers":') or line.startswith('RULE_NUMBERS:'):
            rule_numbers_text = line.split(':', 1)[1].strip()
            # Clean up the text to extract the list
            rule_numbers_text = rule_numbers_text.strip('[]," \t')
            result["rule_numbers"] = [r.strip(' ",') for r in rule_numbers_text.split(',')]
        elif line.startswith('"keywords":') or line.startswith('KEYWORDS:'):
            keywords_text = line.split(':', 1)[1].strip()
            # Clean up the text to extract the list
            keywords_text = keywords_text.strip('[]," \t')
            result["keywords"] = [k.strip(' ",') for k in keywords_text.split(',')]
    
    return result

# Example usage
if __name__ == "__main__":
    # Example chunk from the rules
    input_chunk = """
    EV.5.1.2 Each Accumulator Segment must contain: Maximum static voltage of less than 120 V DC  Maximum energy of 6 MJ The contained energy of a stack is calculated by multiplying the maximum stack voltage with the nominal capacity of the used cell(s). EV.5.1.3 No further energy storage except for reasonably sized intermediate circuit capacitors are allowed after the Energy Meter EV.3.1 EV.5.1.4 All Accumulator Segments and/or Accumulator Containers (including spares and replacement parts) must be identical to the design documented in the ESF and SES EV.5.2 Electrical Configuration EV.5.2.1 All Tractive System components must be rated for the maximum Tractive System voltage EV.5.2.2 If the Accumulator Container is made from an electrically conductive material: a. The poles of the Accumulator Segment(s) and/or cells must be isolated from the inner wall of the Accumulator Container with an insulating material that is rated for the maximum Tractive System voltage. b. All conductive surfaces on the outside of the Accumulator Container must have a low resistance connection to the GLV System Ground, see EV.6.7 c. Any conductive penetrations, such as mounting hardware, must be protected against puncturing the insulating barrier. EV.5.2.3 Each Accumulator Segment must be electrically insulated with suitable Nonflammable Material (F.1.18) (not air) for the two: a. Between the segments in the container b. On top of the segment The intent is to prevent arc flashes caused by inter segment contact or by parts/tools accidentally falling into the container during maintenance for example.
    """
    
    result = extract_keywords_and_rules(input_chunk)
    print("Extracted Information:", result)
    print("Rule Numbers:", result["rule_numbers"])
    print("Keywords:", result["keywords"])