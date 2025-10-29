"""
Code Generation Module for Ravan Quantum-ML System
Task 17.1: Implement code generation

Generates Python/Qiskit scripts from task descriptions with syntax validation
"""

import sys
import os
import ast
import re
from typing import Dict, Any, Optional, List
import logging

from src.llm.llm_adapter import LLMAdapter
from src.llm.prompt_manager import PromptManager
from src.llm.code_sandbox import CodeSandbox, validate_code_safety


class CodeGenerator:
    """
    Generates executable Python/Qiskit code from natural language descriptions
    
    Features:
    - Task description → Python code
    - Syntax validation before returning
    - Safe code patterns only
    - Qiskit/NumPy/SciPy focus
    """
    
    def __init__(self, llm_adapter: LLMAdapter, prompt_manager: Optional[PromptManager] = None, use_sandbox: bool = True):
        """
        Initialize code generator
        
        Args:
            llm_adapter: LLMAdapter instance for code generation
            prompt_manager: Optional PromptManager for validation/caching
            use_sandbox: Whether to validate code in sandbox before returning
        """
        self.llm = llm_adapter
        self.prompt_mgr = prompt_manager
        self.use_sandbox = use_sandbox
        self.sandbox = CodeSandbox() if use_sandbox else None
        self.logger = logging.getLogger('ravan.code_generator')
        
        # Allowed imports for safety
        self.allowed_imports = {
            'qiskit', 'qiskit.circuit', 'qiskit.quantum_info',
            'numpy', 'np', 'scipy', 'matplotlib', 'plt',
            'math', 'cmath', 'typing'
        }
        
        # Blocked patterns for safety
        self.blocked_patterns = [
            r'import\s+os',
            r'import\s+subprocess',
            r'import\s+sys',
            r'__import__',
            r'eval\s*\(',
            r'exec\s*\(',
            r'compile\s*\(',
            r'open\s*\(',
            r'file\s*\(',
        ]
    
    def generate_code(
        self,
        task_description: str,
        code_type: str = 'quantum_circuit',
        include_comments: bool = True,
        include_example: bool = True
    ) -> str:
        """
        Generate Python code from task description
        
        Args:
            task_description: Natural language description of task
            code_type: Type of code ('quantum_circuit', 'simulation', 'analysis')
            include_comments: Whether to include explanatory comments
            include_example: Whether to include usage example
            
        Returns:
            Generated Python code
            
        Raises:
            ValueError: If code validation fails
            
        Example:
            >>> generator.generate_code(
            ...     "Create a Bell state circuit with measurement"
            ... )
            '''
            from qiskit import QuantumCircuit
            
            # Create Bell state circuit
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure([0, 1], [0, 1])
            '''
        """
        self.logger.info(f"Generating {code_type} code for: {task_description}")
        
        # Create prompt
        prompt = self._create_code_prompt(
            task_description,
            code_type,
            include_comments,
            include_example
        )
        
        # Query LLM (with prompt manager if available)
        if self.prompt_mgr:
            code = self.prompt_mgr.process_prompt(
                prompt,
                lambda p: self.llm.query(p, max_new_tokens=800, temperature=0.3)
            )
        else:
            code = self.llm.query(prompt, max_new_tokens=800, temperature=0.3)
        
        # Extract code from response
        code = self._extract_code(code)
        
        # Validate code
        is_valid, error = self.validate_code(code)
        if not is_valid:
            raise ValueError(f"Generated code validation failed: {error}")
        
        # Additional sandbox validation if enabled
        if self.use_sandbox:
            is_safe, safety_error = validate_code_safety(code)
            if not is_safe:
                raise ValueError(f"Generated code failed sandbox validation: {safety_error}")
            self.logger.info("Code passed sandbox safety validation")
        
        self.logger.info("Code generated and validated successfully")
        return code
    
    def _create_code_prompt(
        self,
        task_description: str,
        code_type: str,
        include_comments: bool,
        include_example: bool
    ) -> str:
        """Create prompt for code generation"""
        
        if code_type == 'quantum_circuit':
            return self._create_quantum_circuit_prompt(
                task_description, include_comments, include_example
            )
        elif code_type == 'simulation':
            return self._create_simulation_prompt(
                task_description, include_comments, include_example
            )
        elif code_type == 'analysis':
            return self._create_analysis_prompt(
                task_description, include_comments, include_example
            )
        else:
            raise ValueError(f"Unknown code type: {code_type}")
    
    def _create_quantum_circuit_prompt(
        self,
        task_description: str,
        include_comments: bool,
        include_example: bool
    ) -> str:
        """Create prompt for quantum circuit code"""
        
        comments_instruction = "Include clear comments explaining each step." if include_comments else ""
        example_instruction = "Include a usage example at the end." if include_example else ""
        
        prompt = f"""You are a quantum computing expert. Generate Python code using Qiskit for this task:

Task: {task_description}

Requirements:
- Use Qiskit library (from qiskit import QuantumCircuit, etc.)
- Write clean, executable Python code
- {comments_instruction}
- {example_instruction}
- Use only safe operations (no file I/O, no system calls)
- Return ONLY the Python code, no explanations

Allowed imports: qiskit, numpy, matplotlib
Format: Provide code in a Python code block."""
        
        return prompt
    
    def _create_simulation_prompt(
        self,
        task_description: str,
        include_comments: bool,
        include_example: bool
    ) -> str:
        """Create prompt for simulation code"""
        
        comments_instruction = "Include clear comments." if include_comments else ""
        example_instruction = "Include usage example." if include_example else ""
        
        prompt = f"""You are a quantum physics expert. Generate Python code for this simulation task:

Task: {task_description}

Requirements:
- Use NumPy and SciPy for numerical computations
- Write clean, executable Python code
- {comments_instruction}
- {example_instruction}
- Use only safe operations
- Return ONLY the Python code

Allowed imports: numpy, scipy, matplotlib
Format: Provide code in a Python code block."""
        
        return prompt
    
    def _create_analysis_prompt(
        self,
        task_description: str,
        include_comments: bool,
        include_example: bool
    ) -> str:
        """Create prompt for analysis code"""
        
        comments_instruction = "Include clear comments." if include_comments else ""
        example_instruction = "Include usage example." if include_example else ""
        
        prompt = f"""You are a data analysis expert. Generate Python code for this analysis task:

Task: {task_description}

Requirements:
- Use NumPy, SciPy, Matplotlib for analysis
- Write clean, executable Python code
- {comments_instruction}
- {example_instruction}
- Use only safe operations
- Return ONLY the Python code

Allowed imports: numpy, scipy, matplotlib
Format: Provide code in a Python code block."""
        
        return prompt
    
    def _extract_code(self, response: str) -> str:
        """
        Extract Python code from LLM response
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted code
        """
        # Try to find code in markdown code blocks
        code_block_pattern = r'```python\n(.*?)\n```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Try generic code blocks
        code_block_pattern = r'```\n(.*?)\n```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks, return the whole response (cleaned)
        return response.strip()
    
    def validate_code(self, code: str) -> tuple[bool, Optional[str]]:
        """
        Validate generated code for safety and syntax
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Code contains blocked pattern: {pattern}"
        
        # Check syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        # Check imports
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not self._is_allowed_import(alias.name):
                        return False, f"Disallowed import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and not self._is_allowed_import(node.module):
                    return False, f"Disallowed import: {node.module}"
        
        return True, None
    
    def _is_allowed_import(self, module_name: str) -> bool:
        """Check if import is allowed"""
        # Check if module or its parent is in allowed list
        parts = module_name.split('.')
        for i in range(len(parts)):
            partial = '.'.join(parts[:i+1])
            if partial in self.allowed_imports:
                return True
        return False
    
    def generate_batch(
        self,
        task_descriptions: List[str],
        code_type: str = 'quantum_circuit'
    ) -> List[Optional[str]]:
        """
        Generate code for multiple tasks
        
        Args:
            task_descriptions: List of task descriptions
            code_type: Type of code to generate
            
        Returns:
            List of generated code (None for failed generations)
        """
        results = []
        
        for desc in task_descriptions:
            try:
                code = self.generate_code(desc, code_type)
                results.append(code)
            except Exception as e:
                self.logger.error(f"Failed to generate code for '{desc}': {e}")
                results.append(None)
        
        return results
    
    def add_documentation(self, code: str, description: str) -> str:
        """
        Add docstring documentation to generated code
        
        Args:
            code: Python code
            description: Description of what code does
            
        Returns:
            Code with documentation
        """
        docstring = f'"""\n{description}\n\nGenerated by Ravan Quantum-ML System\n"""\n\n'
        return docstring + code
    
    def format_code(self, code: str) -> str:
        """
        Format code for better readability
        
        Args:
            code: Python code
            
        Returns:
            Formatted code
        """
        # Basic formatting (in production, would use black or autopep8)
        lines = code.split('\n')
        formatted_lines = []
        
        for line in lines:
            # Remove trailing whitespace
            line = line.rstrip()
            formatted_lines.append(line)
        
        # Remove excessive blank lines
        result = []
        prev_blank = False
        for line in formatted_lines:
            is_blank = len(line.strip()) == 0
            if not (is_blank and prev_blank):
                result.append(line)
            prev_blank = is_blank
        
        return '\n'.join(result)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_code_generator(
    llm_adapter: LLMAdapter,
    use_prompt_manager: bool = True
) -> CodeGenerator:
    """
    Create code generator with optional prompt manager
    
    Args:
        llm_adapter: LLMAdapter instance
        use_prompt_manager: Whether to use prompt manager
        
    Returns:
        CodeGenerator instance
    """
    prompt_mgr = None
    if use_prompt_manager:
        from src.llm.prompt_manager import get_prompt_manager
        prompt_mgr = get_prompt_manager()
    
    return CodeGenerator(llm_adapter, prompt_mgr)


def quick_generate(
    task_description: str,
    llm_adapter: LLMAdapter,
    code_type: str = 'quantum_circuit'
) -> str:
    """
    Quick code generation
    
    Args:
        task_description: Task description
        llm_adapter: LLMAdapter instance
        code_type: Type of code
        
    Returns:
        Generated code
    """
    generator = CodeGenerator(llm_adapter)
    return generator.generate_code(task_description, code_type)
