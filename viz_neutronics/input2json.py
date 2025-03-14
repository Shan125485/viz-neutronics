import re
import json
def parse_text_to_dict(filename):
    data_dict = {}  # Root dictionary
    with open(filename, 'r') as file:
        lines = file.readlines()

    stack = [data_dict]  # Stack to keep track of the current dictionary level

    for line in lines:
        line = line.strip()

        # Skip empty lines or comments
        if not line or line.startswith('//'):
            continue

        # Case 1: Inline section with data `{ ... }` on the same line
        if '{' in line and '}' in line:
            section_name, content = line.split('{', 1)
            section_name = section_name.strip()
            new_section = {}
            stack[-1][section_name] = new_section  # Add to current dictionary level
            
            # Process inline content within `{...}`
            inline_content = content.split('}')[0]
            for item in inline_content.split(';'):
                if item.strip():
                    k, v = item.split(maxsplit=1)
                    new_section[k.strip()] = v.strip()
            continue  # Move to the next line

        # Case 2: Start of a new section, no inline data
        elif '{' in line:
            section_name = line.split('{')[0].strip()
            new_section = {}
            stack[-1][section_name] = new_section  # Add to current dictionary level
            stack.append(new_section)  # Push new section onto stack
            continue

        # Case 3: End of a section
        elif '}' in line:
            # Pop the stack to close the section at every `}`
            if len(stack) > 1:  # Prevents popping the root dictionary
                stack.pop()
            continue

        # Case 4: Simple key-value pair (e.g., `pop 50000;`)
        elif ';' in line:
            parts = line.split(';')[0].split(maxsplit=1)
            key = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else None

            # Convert numeric values if possible
            if value is not None:
                try:
                    value = float(value) if '.' in value else int(value)
                except ValueError:
                    pass
            stack[-1][key] = value

    return data_dict

# Convert the parsed dictionary to JSON and save it
def save_to_json(data, json_filename):
    with open(json_filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Read the JSON back in as a dictionary
def read_json_as_dict(json_filename):
    with open(json_filename, 'r') as json_file:
        data = json.load(json_file)
    return data

def stringTuple_to_array(stringTuple):
    
    stringTuple = stringTuple.replace("(", "")
    stringTuple = stringTuple.replace(")", "")
    stringTuple = stringTuple.replace(",", " ")
    stringTuple = stringTuple.split()
    array = list(map(float, stringTuple))
    return array

def dict2obj(dict1):
    # declaring a class
    class obj:
        
        # constructor
        def __init__(self, dict1):
            self.__dict__.update(dict1)
     
    # using json.loads method and passing json.dumps
    # method and custom object hook as arguments
    return json.loads(json.dumps(dict1), object_hook=obj)



# Example usage
# text_file = 'case1_template.txt'  # replace with your filename
# json_file = 'case1_template.json'

# # Step 1: Parse the text file
# data_dict = parse_text_to_dict(text_file)

# # Step 2: Save the dictionary as a JSON file
# save_to_json(data_dict, json_file)

# # Step 3: Read the JSON file back as a dictionary to verify
# json_data = read_json_as_dict(json_file)
# print(json_data)
# print(json_data.get('inactiveTally'))