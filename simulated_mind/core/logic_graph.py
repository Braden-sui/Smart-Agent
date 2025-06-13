"""
Defines the LogicGraph class for loading and representing planner graphs from YAML.
"""
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

# Assuming logic_primitives.py is in the same directory or accessible in PYTHONPATH
# from .logic_primitives import PrimitiveArgs # If we want to type-check args more strictly

@dataclass
class GraphNode:
    """Represents a single node in the logic graph."""
    node_id: str
    primitive: str
    args: Any # Could be Dict[str, Any] or List[Any] depending on primitive
    next_node: Optional[str] = None
    # For COND primitive
    then_node_id: Optional[str] = None
    else_node_id: Optional[str] = None
    # For SEQ/PAR, args will contain the list of child node IDs

    def __post_init__(self):
        # Basic validation for COND node structure if primitive is COND
        if self.primitive.upper() == "COND":
            if not self.then_node_id:
                # then_node_id is typically part of args for COND in our primitive stub
                # This check is more for conceptual integrity if we move it out of args
                pass # Args validation will be handled by the primitive itself

class LogicGraph:
    """
    Represents a loaded logic graph and provides methods to access its components.
    """
    def __init__(self, graph_id: str, description: str, entry_node_id: str, nodes: Dict[str, GraphNode]):
        self.graph_id = graph_id
        self.description = description
        self.entry_node_id = entry_node_id
        self.nodes: Dict[str, GraphNode] = nodes

        if self.entry_node_id not in self.nodes:
            raise ValueError(f"Entry node ID '{self.entry_node_id}' not found in graph nodes.")

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Retrieves a node by its ID."""
        return self.nodes.get(node_id)

    @classmethod
    def load_from_yaml(cls, file_path: str) -> 'LogicGraph':
        """Loads a logic graph from a YAML file."""
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Graph YAML file not found: {file_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {file_path}: {e}")

        if not isinstance(data, dict):
            raise ValueError(f"Invalid graph YAML format in {file_path}: root should be a dictionary.")

        graph_id = data.get('graph_id', file_path) # Default to file_path if no id
        description = data.get('description', '')
        entry_node_id = data.get('entry_node')
        raw_nodes_data = data.get('nodes')

        if not entry_node_id:
            raise ValueError(f"Missing 'entry_node' in graph YAML: {file_path}")
        if not raw_nodes_data or not isinstance(raw_nodes_data, dict):
            raise ValueError(f"Missing or invalid 'nodes' section in graph YAML: {file_path}")

        parsed_nodes: Dict[str, GraphNode] = {}
        for node_id, node_data in raw_nodes_data.items():
            if not isinstance(node_data, dict):
                raise ValueError(f"Node '{node_id}' in {file_path} must be a dictionary.")
            
            primitive = node_data.get('primitive')
            args = node_data.get('args')
            next_node = node_data.get('next_node')
            
            if not primitive:
                raise ValueError(f"Node '{node_id}' in {file_path} is missing 'primitive'.")
            # Args can be legitimately None or empty for some primitives, so no strict check here yet.

            # Specific handling for COND node structure if we enforce it at graph level
            then_node = None
            else_node = None
            if primitive.upper() == "COND":
                # In our current primitive, then/else are part of 'args'. 
                # If we wanted them as top-level node attributes, we'd parse them here.
                # e.g., then_node = node_data.get('then_node_id')
                # else_node = node_data.get('else_node_id')
                pass # Handled by primitive

            parsed_nodes[node_id] = GraphNode(
                node_id=node_id,
                primitive=primitive,
                args=args,
                next_node=next_node,
                then_node_id=then_node, # Will be None unless parsed specifically
                else_node_id=else_node  # Will be None unless parsed specifically
            )
        
        return cls(graph_id, description, entry_node_id, parsed_nodes)


