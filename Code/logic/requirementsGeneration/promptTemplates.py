"""
Auxiliary file for handling requirements analysis prompt templates.
Contains templates for software requirements generation and unification.
"""


def get_generation_prompt(app_description: str, comments: str) -> str:
    """
    Returns the formatted requirements generation prompt.

    Args:
        app_description (str): Description of the application context
        comments (str): User comments to be analyzed

    Returns:
        str: Complete and formatted generation prompt
    """

    template = """## ROLE
Act as a Senior Requirements Engineer and Systems Analyst specializing in NLP. Your goal is precision, relevance, and excellence through careful analysis and refinement.

## PRIMARY TASK
Analyze the provided user comments and application context to generate a final, polished set of clear, actionable software requirements. This involves synthesizing feedback, filtering by impact, and performing a final self-correction pass to consolidate and perfect the requirements before output.

## INPUT DATA
APPLICATION CONTEXT:
{app_description}

USER COMMENTS:
{comments}

## CRITICAL RULES FOR REQUIREMENTS
You must strictly follow these rules in order:

1.	Synthesize or Discard: Your primary strategy is to synthesize. If multiple comments report a similar issue (e.g., app is slow, login fails), create one single requirement for the core problem, citing all sources. If a comment is isolated, ignore it unless it describes a high-impact issue (crash, security flaw, failure of a core function).

2.	Distinguish FR vs. NFR:
	• Functional (FR): WHAT the system does (features, actions).
	• Non-Functional (NFR): HOW the system performs (speed, security, usability).

3.	Be Specific & Measurable: Requirements must be testable. For qualities like speed or reliability, always include a metric (e.g., "in under 2 seconds", "with 99.9% uptime").

4.	Be Atomic: Each requirement must describe only one capability or constraint.

5.	Self-Correction and Refinement (Final Check): After mentally drafting the requirements based on the rules above, you must perform one final review pass before generating the output. In this pass:
	• Consolidate Further: Scrutinize your list for any remaining overlaps. If two requirements are still too similar (e.g., NFR-A: The system shall load the profile page in <2s and NFR-B: The system shall load the settings page in <2s), merge them into a single, more general requirement (e.g., NFR-C: The system shall load all user-specific account pages in <2s).
	• Re-Verify Classification: For every single requirement on your final list, double-check if its FR/NFR classification is correct.

## REQUIRED OUTPUT FORMAT (COMPACT)
The output must have two sections: Functional Requirements (FR) and Non-Functional Requirements (NFR). Each requirement must be a single line.

• FR Format:
**FR[NNN]:** The system shall [action/feature]. (Based on comments: [List of comment numbers])

• NFR Format:
**NFR[NNN] ([Type]):** The system shall [quality/constraint]. (Based on comments: [List of comment numbers])

(Note: [Type] must be a classification like Performance, Usability, Reliability, etc.)

## EXAMPLES FOR GUIDANCE
• Example of Synthesis: Assume comments 5 & 12 complain about slowness.
NFR001 (Performance): The system shall complete its initial startup in under 4 seconds. (Based on comments: 5, 12)

• Example of High-Impact Single Comment: Assume comment 9 says "app crashed on upload."
NFR002 (Reliability): The system shall handle video uploads up to 500MB with a success rate of 99.5%. (Based on comments: 9)

## TASK KICK-OFF
I will now process the input data, applying all critical rules for synthesis, impact assessment, and the crucial final self-correction pass. My output will be a single, consolidated list of final requirements."""

    return template.format(app_description=app_description, comments=comments)


def get_unification_prompt(app_description: str, requirements: str) -> str:
    """
    Returns the formatted requirements unification prompt.

    Args:
        app_description (str): Description of the application context
        requirements (str): Requirements from different clusters to be unified

    Returns:
        str: Complete and formatted unification prompt
    """

    template = """### ROLE
Act as a Lead Systems Analyst and Requirements Manager. Your expertise lies in reviewing requirements documentation from multiple sources, identifying overlaps, and consolidating them into a single, coherent, and de-duplicated master requirements list. Your work ensures engineering efforts are focused and efficient.

### PRIMARY TASK
You will receive a document containing multiple sets of software requirements, each set generated from a different "cluster" of user feedback. Your task is to analyze all requirements across all clusters to identify and merge similar or duplicate requirements. The final output must be a single, unified list of requirements that is free of redundancy, while carefully preserving all unique information and traceability.

### INPUT DATA FORMAT
APPLICATION CONTEXT:
{app_description}

REQUIREMENTS:
{requirements}

### CORE LOGIC FOR CONSOLIDATION
You must follow this consolidation process meticulously:

1.	Identify Similar Requirements: Scan across all clusters to find requirements that address the same core functionality or quality attribute. "Similar" means:
	• They describe the same user action or system feature, even with different wording.
	• They describe a quality constraint (e.g., performance) for the same or closely related parts of the application.
	• One is a specific instance of another, more general requirement.

2.	Execute the Merge: When you find two or more similar requirements, you must merge them into one new requirement by following these rules:
	• Description: Create a new, more comprehensive description.
		- If the requirements are nearly identical, simply use the clearest phrasing.
		- If they describe related specifics (e.g., one mentions "profile page," another "settings page"), abstract to a more general term (e.g., "user account pages").
		- If they have different metrics (e.g., one says <2s and another <3s), always adopt the stricter metric (in this case, <2s) to ensure all original needs are met.
	• Traceability (Based on comments): This is critical. The new, merged requirement's comment list must be the union of all comment lists from the original requirements. Combine all numbers and remove duplicates.
	• Classification: The merged requirement should retain the original classification (FR or NFR) and NFR Type (e.g., Performance, Usability). Similar requirements should share the same type.

3.	Preserve Unique Requirements: If a requirement from a cluster has no similar counterpart in any other cluster, it is unique. It must be preserved as-is and copied directly to the final list.

4.	Re-Numbering: The final, consolidated list of requirements (both FR and NFR) must be re-numbered sequentially starting from 001 for each category.

### REQUIRED OUTPUT FORMAT
Your final output must be a single, clean list, structured into two sections. It should NOT mention the original clusters, as the goal is a unified list.

• Functional Requirements (FR):
**FR[NNN]:** [Consolidated description]. (Based on comments: [Combined list of numbers])

• Non-Functional Requirements (NFR):
**NFR[NNN] ([Type]):** [Consolidated description]. (Based on comments: [Combined list of numbers])

### EXAMPLE OF CONSOLIDATION
INPUT DOCUMENT:
## CLUSTER A: Login & Profile Issues ###
**FR001:** The system shall allow users to reset their password via email. (Based on comments: 4, 15)
**NFR001 (Performance):** The system shall load the user's profile page in under 3 seconds. (Based on comments: 7, 22)

## CLUSTER B: General Performance Complaints ###
**NFR001 (Performance):** The system shall load pages quickly. (Based on comments: 3, 9)
**NFR002 (Performance):** The system shall ensure the account dashboard loads in less than 2 seconds. (Based on comments: 18)

## CLUSTER C: Account Management ###
**FR001:** The system shall provide a 'Forgot Password' link on the login screen. (Based on comments: 2, 11)

CORRECT FINAL OUTPUT:
Functional Requirements (FR)
FR001: The system shall allow users to reset their password via an email link accessible from the login screen. (Based on comments: 2, 4, 11, 15)

Non-Functional Requirements (NFR)
NFR001 (Performance): The system shall load all user account pages (including profile and dashboard) in under 2 seconds. (Based on comments: 3, 7, 9, 18, 22)

### TASK KICK-OFF
I will now process the provided multi-cluster requirements document. I will apply the core logic for consolidation to produce a single, de-duplicated, and master list of requirements in the specified format."""

    return template.format(app_description=app_description, requirements=requirements)


# Usage example
if __name__ == "__main__":
    # Example for generation prompt
    app_desc = "Task management system for development teams"
    user_comments = "1. The app is very slow\n2. Cannot edit tasks\n3. Missing notifications"

    generation_prompt = get_generation_prompt(app_desc, user_comments)
    print("=== GENERATION PROMPT ===")
    print(generation_prompt[:200] + "...")

    print("\n" + "=" * 50 + "\n")

    # Example for unification prompt
    reqs = """## CLUSTER A:
**FR001:** System should allow creating tasks. (Based on comments: 1, 3)
## CLUSTER B:
**FR001:** User can add new tasks. (Based on comments: 2, 5)"""

    unification_prompt = get_unification_prompt(app_desc, reqs)
    print("=== UNIFICATION PROMPT ===")
    print(unification_prompt[:200] + "...")