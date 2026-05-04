import json
import re

print("📚 Preparing training data from your OS notes...")

# Read your OS notes
with open("../week-01-embeddings/my_os_notes.txt", "r", encoding="utf-8") as f:
    notes = f.read()

# Split into sections (each heading is a potential Q&A)
sections = re.split(r'\n#{1,3} ', notes)

training_pairs = []

for section in sections:
    if not section.strip():
        continue
    
    # Extract the heading as the question
    lines = section.strip().split('\n')
    heading = lines[0].strip()
    
    # Extract the content as the answer
    content = '\n'.join(lines[1:]) if len(lines) > 1 else ""
    
    if heading and content and len(content) > 50:  # Only substantial content
        training_pairs.append({
            "instruction": f"Explain {heading}",
            "output": content[:1000]  # Limit length
        })
        
        # Also add a generic question version
        training_pairs.append({
            "instruction": f"What is {heading}?",
            "output": content[:1000]
        })

print(f"✅ Created {len(training_pairs)} training pairs")

# Save to JSON
with open("training_data.json", "w", encoding="utf-8") as f:
    json.dump(training_pairs, f, indent=2)

print(f"📁 Saved to training_data.json")
print(f"📊 Examples:")
for i in range(min(3, len(training_pairs))):
    print(f"\n{i+1}. Q: {training_pairs[i]['instruction']}")
    print(f"   A: {training_pairs[i]['output'][:100]}...")