from typing import Dict, List

TEMPLATES: Dict[str, List[str]] = {
    "understand project": [
        "Review project documentation and README.",
        "Analyze main components and their interactions.",
        "Identify key data structures and flows.",
        "List potential areas for improvement or extension."
    ],
    "develop feature": [
        "Clarify feature requirements and acceptance criteria.",
        "Design the feature, including UI/UX if applicable.",
        "Implement the core logic for the feature.",
        "Write unit and integration tests for the feature.",
        "Document the new feature."
    ],
    "understand the agent’s planning structure": [
        "Review the Planner class in planner.py.",
        "Analyze how goals are decomposed into sub-tasks.",
        "Investigate the role of planner_rules.py and templates.",
        "Examine how memory (MemoryStore) is used for suggestions or caching plans.",
        "Understand the recursive planning mechanism and MAX_RECURSION_DEPTH."
    ],
    "develop recursive decomposition": [
        "Define the Goal data structure (id, description, priority, sub_goals, parent_goal).",
        "Implement recursive_plan(goal, depth) in Planner.",
        "Ensure base cases for recursion (max depth, or goal is atomic).",
        "Integrate template-based decomposition within recursion.",
        "Integrate memory-based suggestions within recursion.",
        "Test recursive planning with various complex goals."
    ]
}


# --- Graph-Based Planning Templates ---
# Maps goal keywords/types to their corresponding graph YAML file paths.
# The LogicEngine will use these graphs to decompose goals.
GRAPH_TEMPLATES = {
    "develop feature": "simulated_mind/templates/graphs/develop_feature.yaml",
    # Example for another potential graph:
    # "understand project": "simulated_mind/templates/graphs/understand_project.yaml",
}

