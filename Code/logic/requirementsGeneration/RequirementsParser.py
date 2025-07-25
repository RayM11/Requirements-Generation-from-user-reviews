"""
Requirements Parser Module

This module handles the parsing of LLM responses to extract structured requirements
information for both generation and unification processes.
"""

import re
from typing import List, Dict, Tuple


class RequirementsParser:
    """
    Parser for extracting structured requirements from LLM responses.

    Handles both generation responses (requirements by cluster) and
    unification responses (consolidated requirements).
    """

    def __init__(self):
        """Initialize the RequirementsParser."""
        # Regex patterns for parsing requirements
        self.fr_pattern = r'\*\*FR(\d+):\*\* (.+?) \(Based on comments: ([^)]+)\)'
        self.nfr_pattern = r'\*\*NFR(\d+) \(([^)]+)\):\*\* (.+?) \(Based on comments: ([^)]+)\)'

        # Alternative patterns for more flexible matching
        self.fr_pattern_alt = r'FR(\d+):\s*(.+?)\s*\(Based on comments:\s*([^)]+)\)'
        self.nfr_pattern_alt = r'NFR(\d+)\s*\(([^)]+)\):\s*(.+?)\s*\(Based on comments:\s*([^)]+)\)'

    def parse_generation_response(self, response: str) -> List[Dict[str, str]]:
        """
        Parse LLM response from requirements generation to extract structured requirements.

        Args:
            response (str): Raw LLM response containing requirements

        Returns:
            List[Dict[str, str]]: List of parsed requirements with structure:
                - type: 'FUNCTIONAL' or 'NON_FUNCTIONAL'
                - description: Requirement description text
                - based_on_comments: Comma-separated comment numbers
                - nfr_type: Type of NFR (only for non-functional requirements)
        """
        requirements = []

        # Clean the response
        response = self._clean_response(response)

        # Parse functional requirements
        fr_requirements = self._parse_functional_requirements(response)
        requirements.extend(fr_requirements)

        # Parse non-functional requirements
        nfr_requirements = self._parse_non_functional_requirements(response)
        requirements.extend(nfr_requirements)

        if not requirements:
            print("Warning: No requirements found in response. Attempting alternative parsing...")
            # Try alternative parsing methods
            requirements = self._parse_alternative_format(response)

        return requirements

    def parse_unification_response(self, response: str) -> List[Dict[str, str]]:
        """
        Parse LLM response from requirements unification to extract consolidated requirements.

        Args:
            response (str): Raw LLM response containing unified requirements

        Returns:
            List[Dict[str, str]]: List of parsed unified requirements
        """
        requirements = []

        # Clean the response
        response = self._clean_response(response)

        # Parse functional requirements
        fr_requirements = self._parse_functional_requirements(response)
        requirements.extend(fr_requirements)

        # Parse non-functional requirements
        nfr_requirements = self._parse_non_functional_requirements(response)
        requirements.extend(nfr_requirements)

        if not requirements:
            print("Warning: No unified requirements found in response. Attempting alternative parsing...")
            requirements = self._parse_alternative_format(response)

        return requirements

    def _clean_response(self, response: str) -> str:
        """
        Clean and preprocess the LLM response for parsing.

        Args:
            response (str): Raw response text

        Returns:
            str: Cleaned response text
        """
        # Remove extra whitespace and normalize line endings
        response = re.sub(r'\n\s*\n', '\n', response)
        response = re.sub(r'\s+', ' ', response)

        # Remove common markdown formatting that might interfere
        response = response.replace('```', '')
        response = response.replace('###', '')
        response = response.replace('##', '')

        return response.strip()

    def _parse_functional_requirements(self, response: str) -> List[Dict[str, str]]:
        """
        Parse functional requirements from response.

        Args:
            response (str): Cleaned response text

        Returns:
            List[Dict[str, str]]: List of functional requirements
        """
        requirements = []

        # Try primary pattern first
        matches = re.findall(self.fr_pattern, response, re.DOTALL | re.IGNORECASE)

        if not matches:
            # Try alternative pattern
            matches = re.findall(self.fr_pattern_alt, response, re.DOTALL | re.IGNORECASE)

        for match in matches:
            if len(match) == 3:
                req_id, description, comments = match
                requirements.append({
                    'requirement_id': f'FR{req_id.zfill(3)}',
                    'type': 'FUNCTIONAL',
                    'description': description.strip(),
                    'based_on_comments': self._clean_comment_list(comments),
                    'nfr_type': None
                })

        return requirements

    def _parse_non_functional_requirements(self, response: str) -> List[Dict[str, str]]:
        """
        Parse non-functional requirements from response.

        Args:
            response (str): Cleaned response text

        Returns:
            List[Dict[str, str]]: List of non-functional requirements
        """
        requirements = []

        # Try primary pattern first
        matches = re.findall(self.nfr_pattern, response, re.DOTALL | re.IGNORECASE)

        if not matches:
            # Try alternative pattern
            matches = re.findall(self.nfr_pattern_alt, response, re.DOTALL | re.IGNORECASE)

        for match in matches:
            if len(match) == 4:
                req_id, nfr_type, description, comments = match
                requirements.append({
                    'requirement_id': f'NFR{req_id.zfill(3)}',
                    'type': 'NON_FUNCTIONAL',
                    'description': description.strip(),
                    'based_on_comments': self._clean_comment_list(comments),
                    'nfr_type': nfr_type.strip()
                })

        return requirements

    def _parse_alternative_format(self, response: str) -> List[Dict[str, str]]:
        """
        Alternative parsing method for responses that don't match standard patterns.

        Args:
            response (str): Response text to parse

        Returns:
            List[Dict[str, str]]: List of parsed requirements
        """
        requirements = []

        # Split response into lines and look for requirement-like patterns
        lines = response.split('\n')

        fr_counter = 1
        nfr_counter = 1

        for line in lines:
            line = line.strip()

            # Look for functional requirement indicators
            if any(indicator in line.lower() for indicator in ['fr', 'functional', 'system shall', 'the system']):
                # Try to extract functional requirement
                req = self._extract_requirement_from_line(line, 'FUNCTIONAL', fr_counter)
                if req:
                    requirements.append(req)
                    fr_counter += 1

            # Look for non-functional requirement indicators
            elif any(indicator in line.lower() for indicator in ['nfr', 'performance', 'usability', 'reliability']):
                # Try to extract non-functional requirement
                req = self._extract_requirement_from_line(line, 'NON_FUNCTIONAL', nfr_counter)
                if req:
                    requirements.append(req)
                    nfr_counter += 1

        return requirements

    def _extract_requirement_from_line(self, line: str, req_type: str, counter: int) -> Dict[str, str]:
        """
        Extract requirement information from a single line.

        Args:
            line (str): Line of text to parse
            req_type (str): Type of requirement ('FUNCTIONAL' or 'NON_FUNCTIONAL')
            counter (int): Counter for ID generation

        Returns:
            Dict[str, str]: Parsed requirement or None if no valid requirement found
        """
        # Look for comment references
        comment_match = re.search(r'\(Based on comments?:\s*([^)]+)\)', line, re.IGNORECASE)
        comments = comment_match.group(1) if comment_match else "Unknown"

        # Clean the line to get description
        description = line
        if comment_match:
            description = line[:comment_match.start()].strip()

        # Remove common prefixes and formatting
        description = re.sub(r'^\*\*[^:]*:\*\*\s*', '', description)
        description = re.sub(r'^[A-Z]+\d+\s*[:(]\s*[^)]*\)?\s*:?\s*', '', description)
        description = description.strip()

        # Only return if we have meaningful content
        if len(description) > 10 and 'system' in description.lower():
            prefix = 'FR' if req_type == 'FUNCTIONAL' else 'NFR'
            nfr_type = self._extract_nfr_type_from_line(line) if req_type == 'NON_FUNCTIONAL' else None

            return {
                'requirement_id': f'{prefix}{counter:03d}',
                'type': req_type,
                'description': description,
                'based_on_comments': self._clean_comment_list(comments),
                'nfr_type': nfr_type
            }

        return None

    def _extract_nfr_type_from_line(self, line: str) -> str:
        """
        Extract NFR type from a line of text.

        Args:
            line (str): Line containing NFR information

        Returns:
            str: NFR type
        """
        # Look for explicit type in parentheses
        type_match = re.search(r'\(([^)]+)\)', line)
        if type_match:
            potential_type = type_match.group(1).strip()
            # Check if it looks like an NFR type
            nfr_types = ['Performance', 'Usability', 'Reliability', 'Security', 'Scalability', 'Maintainability']
            for nfr_type in nfr_types:
                if nfr_type.lower() in potential_type.lower():
                    return nfr_type

        # Fallback: analyze content
        return self._infer_nfr_type_from_content(line)

    def _infer_nfr_type_from_content(self, content: str) -> str:
        """
        Infer NFR type from content analysis.

        Args:
            content (str): Content to analyze

        Returns:
            str: Inferred NFR type
        """
        content_lower = content.lower()

        type_keywords = {
            'Performance': ['load', 'speed', 'time', 'fast', 'slow', 'seconds', 'response', 'latency'],
            'Usability': ['user', 'interface', 'navigation', 'ease', 'intuitive', 'accessible', 'friendly'],
            'Reliability': ['crash', 'error', 'failure', 'stability', 'available', 'uptime', 'robust'],
            'Security': ['secure', 'authentication', 'authorization', 'password', 'encrypt', 'protected'],
            'Scalability': ['scale', 'concurrent', 'users', 'capacity', 'volume', 'growth'],
            'Maintainability': ['maintain', 'update', 'modify', 'extend', 'documentation', 'readable']
        }

        for nfr_type, keywords in type_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                return nfr_type

        return 'Quality'  # Default type

    def _clean_comment_list(self, comments: str) -> str:
        """
        Clean and format the comment list.

        Args:
            comments (str): Raw comment string

        Returns:
            str: Cleaned comment list
        """
        # Remove extra whitespace
        comments = comments.strip()

        # Remove common formatting
        comments = re.sub(r'[,\s]+', ', ', comments)
        comments = re.sub(r'^,\s*', '', comments)
        comments = re.sub(r',\s*$', '', comments)

        # Sort numbers if they appear to be numeric
        try:
            comment_numbers = [int(x.strip()) for x in comments.split(',') if x.strip().isdigit()]
            if comment_numbers:
                return ', '.join(map(str, sorted(comment_numbers)))
        except:
            pass

        return comments

    def validate_parsed_requirements(self, requirements: List[Dict[str, str]]) -> Tuple[
        List[Dict[str, str]], List[str]]:
        """
        Validate parsed requirements and return valid ones along with error messages.

        Args:
            requirements (List[Dict[str, str]]): List of parsed requirements

        Returns:
            Tuple[List[Dict[str, str]], List[str]]: Valid requirements and error messages
        """
        valid_requirements = []
        errors = []

        for i, req in enumerate(requirements):
            # Check required fields
            required_fields = ['requirement_id', 'type', 'description', 'based_on_comments']
            missing_fields = [field for field in required_fields if field not in req or not req[field]]

            if missing_fields:
                errors.append(f"Requirement {i + 1}: Missing fields: {missing_fields}")
                continue

            # Check type validity
            if req['type'] not in ['FUNCTIONAL', 'NON_FUNCTIONAL']:
                errors.append(f"Requirement {req['requirement_id']}: Invalid type: {req['type']}")
                continue

            # Check description length
            if len(req['description']) < 10:
                errors.append(f"Requirement {req['requirement_id']}: Description too short")
                continue

            valid_requirements.append(req)

        return valid_requirements, errors


# Example usage and testing
if __name__ == "__main__":
    """
    Test the RequirementsParser with sample responses.
    """

    parser = RequirementsParser()

    # Sample generation response for testing
    sample_generation_response = """
    Functional Requirements (FR)
    **FR001:** The system shall allow users to reset their password via email. (Based on comments: 4, 15)
    **FR002:** The system shall provide a task creation interface. (Based on comments: 1, 3, 7)
    **FR003:** The system shall enable task editing capabilities. (Based on comments: 2, 8)

    Non-Functional Requirements (NFR)
    **NFR001 (Performance):** The system shall load pages in under 2 seconds. (Based on comments: 5, 12, 18)
    **NFR002 (Usability):** The system shall provide intuitive navigation. (Based on comments: 6, 11)
    **NFR003 (Reliability):** The system shall handle file uploads with 99% success rate. (Based on comments: 9, 14)
    """

    # Sample unification response for testing
    sample_unification_response = """
    Functional Requirements (FR)
    FR001: The system shall allow users to reset their password via an email link accessible from the login screen. (Based on comments: 2, 4, 11, 15)
    FR002: The system shall provide comprehensive task management capabilities including creation and editing. (Based on comments: 1, 2, 3, 7, 8)

    Non-Functional Requirements (NFR)
    NFR001 (Performance): The system shall load all user interface pages in under 2 seconds. (Based on comments: 5, 12, 18, 22)
    NFR002 (Reliability): The system shall handle file uploads and downloads with 99.5% success rate. (Based on comments: 9, 14, 20)
    """

    print("=" * 60)
    print("TESTING REQUIREMENTS PARSER")
    print("=" * 60)

    # Test generation response parsing
    print("\n1. Testing Generation Response Parsing:")
    print("-" * 40)

    gen_requirements = parser.parse_generation_response(sample_generation_response)

    print(f"Found {len(gen_requirements)} requirements:")
    for req in gen_requirements:
        print(f"  {req['requirement_id']} ({req['type']}): {req['description'][:50]}...")
        print(f"    Based on comments: {req['based_on_comments']}")
        if req['nfr_type']:
            print(f"    NFR Type: {req['nfr_type']}")

    # Test unification response parsing
    print("\n2. Testing Unification Response Parsing:")
    print("-" * 40)

    unif_requirements = parser.parse_unification_response(sample_unification_response)

    print(f"Found {len(unif_requirements)} unified requirements:")
    for req in unif_requirements:
        print(f"  {req['requirement_id']} ({req['type']}): {req['description'][:50]}...")
        print(f"    Based on comments: {req['based_on_comments']}")
        if req['nfr_type']:
            print(f"    NFR Type: {req['nfr_type']}")

    # Test validation
    print("\n3. Testing Validation:")
    print("-" * 40)

    valid_reqs, errors = parser.validate_parsed_requirements(gen_requirements)
    print(f"Valid requirements: {len(valid_reqs)}")
    print(f"Errors found: {len(errors)}")

    if errors:
        for error in errors:
            print(f"  - {error}")

    # Test alternative format parsing
    print("\n4. Testing Alternative Format Parsing:")
    print("-" * 40)

    alternative_response = """
    The system requirements are:
    - FR: The system shall provide user authentication (Based on comments: 1, 5)
    - NFR Performance: Response time should be under 3 seconds (Based on comments: 2, 7)
    - The system shall allow data export functionality (Based on comments: 3)
    """

    alt_requirements = parser.parse_generation_response(alternative_response)
    print(f"Found {len(alt_requirements)} requirements from alternative format:")
    for req in alt_requirements:
        print(f"  {req['requirement_id']}: {req['description'][:60]}...")

    print("\n" + "=" * 60)
    print("PARSER TESTING COMPLETED")
    print("=" * 60)