import ast
import re
from pathlib import Path

# Configuration
MODULES_TO_DOC = [
    ("src/memmachine/rest_client/client.py", "docs/api_reference/python/client_api.mdx", "Client API", "Low-level API reference for MemMachineClient", "code"),
    ("src/memmachine/rest_client/project.py", "docs/api_reference/python/project_api.mdx", "Project API", "API reference for Project management", "folder"),
    ("src/memmachine/rest_client/memory.py", "docs/api_reference/python/memory_api.mdx", "Memory API", "API reference for Memory operations", "brain"),
    ("src/memmachine/episodic_memory/episodic_memory.py", "docs/api_reference/python/episodic_memory.mdx", "Episodic Memory", "Server-side Episodic Memory API", "server"),
    ("src/memmachine/episodic_memory/episodic_memory_manager.py", "docs/api_reference/python/episodic_memory_manager.mdx", "Episodic Memory Manager", "Manager for Episodic Memory instances", "sitemap"),
]

def get_annotation(arg):
    """Extract string representation of type annotation."""
    if arg.annotation is None:
        return None
    try:
        return ast.unparse(arg.annotation)
    except AttributeError:
        return "Any"

def parse_docstring_google(docstring):
    """
    Parses a Google-style docstring into sections.
    Returns a dict with keys: 'description', 'args', 'returns', 'raises'.
    'args' is a list of dicts: {'name', 'type', 'desc'}.
    """
    if not docstring:
        return {"description": "", "args": [], "returns": "", "raises": ""}

    lines = docstring.split('\n')
    sections = {"description": [], "args": [], "returns": [], "raises": []}
    current_section = "description"
    
    # Regex for args: "name (type): description" or "name: description"
    arg_regex = re.compile(r"^\s*(\w+)(?:\s*([\w\[\]\s,|]+))?:\s*(.+)$")
    
    # Regex for section headers
    section_headers = {
        "Args:": "args",
        "Arguments:": "args",
        "Parameters:": "args",
        "Returns:": "returns",
        "Raises:": "raises",
        "Example:": "example",
        "Examples:": "example"
    }

    for line in lines:
        stripped_line = line.strip()
        
        # Check if this line is a section header
        if stripped_line in section_headers:
            current_section = section_headers[stripped_line]
            continue
            
        if current_section == "description":
            sections["description"].append(line)
            
        elif current_section == "args":
            match = arg_regex.match(line)
            if match:
                name, type_, desc = match.groups()
                sections["args"].append({"name": name, "type": type_ or "", "desc": desc})
            elif sections["args"]:
                # Continuation of previous arg description
                sections["args"][-1]["desc"] += " " + stripped_line
                
        elif current_section == "returns":
            sections["returns"].append(line)
            
        elif current_section == "raises":
            sections["raises"].append(line)

    # Join description lines
    sections["description"] = "\n".join(sections["description"]).strip()
    sections["returns"] = "\n".join(sections["returns"]).strip()
    sections["raises"] = "\n".join(sections["raises"]).strip()
    
    return sections

def format_mdx_method(name, args, docstring_info, is_class=False):
    """Formats a method/function documentation in MDX style."""
    
    output = []
    
    # Header
    if is_class:
        output.append(f"## Class `{name}`")
    else:
        output.append(f"### `{name}`")
    
    output.append("")
    
    # Description
    if docstring_info["description"]:
        output.append(docstring_info["description"])
        output.append("")
    
    # Signature block
    sig_parts = []
    for arg in args:
        arg_name = arg.arg
        if arg_name == 'self': continue
        
        # Try to get type from AST annotation first, then docstring
        type_hint = get_annotation(arg)
        
        # Find matching docstring arg
        doc_arg = next((a for a in docstring_info["args"] if a["name"] == arg_name), None)
        
        if not type_hint and doc_arg and doc_arg["type"]:
            type_hint = doc_arg["type"]
            
        if type_hint:
            sig_parts.append(f"{arg_name}: {type_hint}")
        else:
            sig_parts.append(arg_name)
            
    sig_str = ", ".join(sig_parts)
    output.append(f"```python\ndef {name}({sig_str})\n```")
    output.append("")

    # Parameters Table (Mintlify Style)
    # Merging AST args and Docstring args
    if args or docstring_info["args"]:
        output.append("**Parameters**")
        output.append("")
        
        # Use Mintlify <ParamField> if desired, or standard table. 
        # Let's use a clean Markdown table for compatibility.
        output.append("| Name | Type | Description |")
        output.append("| :--- | :--- | :--- |")
        
        # Iterate over actual AST args to preserve order
        for arg in args:
            arg_name = arg.arg
            if arg_name == 'self': continue
            
            type_hint = get_annotation(arg)
            doc_arg = next((a for a in docstring_info["args"] if a["name"] == arg_name), None)
            
            desc = "-"
            if doc_arg:
                desc = doc_arg["desc"]
                if not type_hint and doc_arg["type"]:
                    type_hint = doc_arg["type"]
            
            # Escape pipes in description
            desc = desc.replace("|", "||")
            
            # Format code ticks
            type_str = f"`{type_hint}`" if type_hint else "-"
            name_str = f"`{arg_name}`"
            
            output.append(f"| {name_str} | {type_str} | {desc} |")
            
        output.append("")

    # Return Value
    if docstring_info["returns"]:
        output.append("**Returns**")
        output.append("")
        output.append(docstring_info["returns"])
        output.append("")

    # Raises
    if docstring_info["raises"]:
        output.append("**Raises**")
        output.append("")
        output.append(docstring_info["raises"])
        output.append("")

    return "\n".join(output)

def generate_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    md_output = []
    
    # Module docstring
    module_doc = ast.get_docstring(tree)
    if module_doc:
        md_output.append(module_doc)
        md_output.append("")

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            # Class Documentation
            class_doc = ast.get_docstring(node)
            doc_info = parse_docstring_google(class_doc)
            
            # Find __init__ for constructor args
            init_method = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
            init_args = init_method.args.args if init_method else []
            
            md_output.append(format_mdx_method(node.name, init_args, doc_info, is_class=True))
            
            # Methods
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name.startswith("_") and item.name != "__init__":
                        continue # Skip private methods
                    if item.name == "__init__":
                        continue # Skip init as it's merged into class doc (or usually is)
                    
                    method_doc = ast.get_docstring(item)
                    method_info = parse_docstring_google(method_doc)
                    
                    md_output.append(format_mdx_method(item.name, item.args.args, method_info))

        elif isinstance(node, ast.FunctionDef):
             if node.name.startswith("_"):
                continue
             
             func_doc = ast.get_docstring(node)
             func_info = parse_docstring_google(func_doc)
             md_output.append(format_mdx_method(node.name, node.args.args, func_info))

    return "\n".join(md_output)

def main():
    root_dir = Path(__file__).parent.parent
    
    for src_rel, dest_rel, title, desc, icon in MODULES_TO_DOC:
        src_path = root_dir / src_rel
        dest_path = root_dir / dest_rel
        
        print(f"Processing {src_path}...")
        
        if not src_path.exists():
            print(f"Warning: {src_path} not found. Skipping.")
            continue
            
        markdown_body = generate_markdown(src_path)
        
        frontmatter = f"""
---
title: "{title}"
description: "{desc}"
icon: "{icon}"
---

"""
        content = frontmatter + markdown_body
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        print(f"Generated {dest_path}")

if __name__ == "__main__":
    main()