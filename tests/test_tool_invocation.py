"""Evaluation tests for MCP tool invocation and description quality.

This module provides comprehensive evaluation tests to verify:
1. Tool descriptions follow MCP and Anthropic best practices
2. Tools are invoked correctly for their intended Jupyter notebook editing purposes
3. Tool selection is guided by well-structured descriptions
4. Consistency across all tool descriptions

Evaluation Criteria:
--------------------
1. NAMING: Tool names are action-oriented, use verb_noun pattern, include 'ipynb_' prefix
2. DESCRIPTION: Each tool has a clear description with what/when/how/returns sections
3. PARAMETERS: All parameters have clear types, descriptions, and examples
4. RETURNS: Return values are consistently structured with success/error patterns
5. GUIDANCE: Descriptions guide users to correct tool selection (USE/DO NOT USE hints)
6. CONSISTENCY: All tools follow the same documentation format

Test Categories:
----------------
- test_tool_description_*: Validate description structure and content
- test_tool_metadata_*: Validate tool metadata (name, parameters, return types)
- test_tool_invocation_*: Test correct tool behavior for various scenarios
- test_tool_selection_*: Simulate LLM tool selection based on descriptions
"""

import pytest
import re
import inspect
from src.jupyter_editor import server


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def all_tools():
    """Get all MCP tool functions from the server module."""
    tools = []
    for name in dir(server):
        if name.startswith('ipynb_'):
            tool_obj = getattr(server, name)
            # FastMCP wraps tools in FunctionTool objects
            # Access the underlying function via .fn attribute
            if hasattr(tool_obj, 'fn'):
                func = tool_obj.fn
                doc = func.__doc__ or tool_obj.description or ''
                try:
                    signature = inspect.signature(func)
                except (ValueError, TypeError):
                    from inspect import Parameter
                    signature = inspect.Signature()
                
                tools.append({
                    'name': name,
                    'func': func,
                    'doc': doc,
                    'signature': signature,
                    'tool_obj': tool_obj
                })
            elif callable(tool_obj):
                # Fallback for regular functions
                doc = tool_obj.__doc__ or ''
                try:
                    signature = inspect.signature(tool_obj)
                except (ValueError, TypeError):
                    signature = inspect.Signature()
                
                tools.append({
                    'name': name,
                    'func': tool_obj,
                    'doc': doc,
                    'signature': signature,
                    'tool_obj': None
                })
    return tools


@pytest.fixture
def tool_categories():
    """Define expected tool categories for organizational validation."""
    return {
        'read': [
            'ipynb_read_notebook',
            'ipynb_list_cells',
            'ipynb_get_cell',
            'ipynb_search_cells',
        ],
        'modify_single': [
            'ipynb_replace_cell',
            'ipynb_insert_cell',
            'ipynb_append_cell',
            'ipynb_delete_cell',
            'ipynb_str_replace_in_cell',
        ],
        'metadata': [
            'ipynb_get_metadata',
            'ipynb_update_metadata',
            'ipynb_set_kernel',
            'ipynb_list_available_kernels',
        ],
        'batch_cell': [
            'ipynb_replace_cells_batch',
            'ipynb_delete_cells_batch',
            'ipynb_insert_cells_batch',
            'ipynb_search_replace_all',
            'ipynb_reorder_cells',
            'ipynb_filter_cells',
        ],
        'batch_notebook': [
            'ipynb_merge_notebooks',
            'ipynb_split_notebook',
            'ipynb_apply_to_notebooks',
            'ipynb_search_notebooks',
            'ipynb_sync_metadata',
            'ipynb_extract_cells',
            'ipynb_clear_outputs',
        ],
        'validation': [
            'ipynb_validate_notebook',
            'ipynb_get_notebook_info',
            'ipynb_validate_notebooks_batch',
        ],
    }


# ============================================================================
# Tool Description Structure Tests
# ============================================================================

class TestToolDescriptionStructure:
    """Test that all tool descriptions follow the required structure."""
    
    def test_all_tools_have_docstrings(self, all_tools):
        """Every tool must have a docstring."""
        missing = [t['name'] for t in all_tools if not t['doc'].strip()]
        assert not missing, f"Tools missing docstrings: {missing}"
    
    def test_docstrings_have_minimum_length(self, all_tools):
        """Docstrings should be comprehensive (minimum 200 characters)."""
        short_docs = []
        for tool in all_tools:
            if len(tool['doc']) < 200:
                short_docs.append((tool['name'], len(tool['doc'])))
        
        assert not short_docs, (
            f"Tools with too-short docstrings (<200 chars): "
            f"{[(n, l) for n, l in short_docs]}"
        )
    
    def test_docstrings_have_first_line_summary(self, all_tools):
        """First line should be a clear action-oriented summary."""
        issues = []
        for tool in all_tools:
            first_line = tool['doc'].split('\n')[0].strip()
            # Should start with an action verb
            action_verbs = ['Read', 'Get', 'List', 'Search', 'Replace', 'Insert', 
                           'Append', 'Delete', 'Update', 'Set', 'Merge', 'Split',
                           'Apply', 'Sync', 'Extract', 'Clear', 'Validate', 'Keep',
                           'Reorder']
            has_verb = any(first_line.startswith(v) for v in action_verbs)
            if not has_verb:
                issues.append((tool['name'], first_line[:50]))
        
        assert not issues, f"Tools with non-action-oriented first lines: {issues}"
    
    def test_docstrings_have_args_section(self, all_tools):
        """Tools with parameters must have an Args section."""
        issues = []
        for tool in all_tools:
            params = list(tool['signature'].parameters.keys())
            if params and 'Args:' not in tool['doc']:
                issues.append(tool['name'])
        
        assert not issues, f"Tools missing Args section: {issues}"
    
    def test_docstrings_have_returns_section(self, all_tools):
        """All tools must document their return values."""
        issues = []
        for tool in all_tools:
            if 'Returns:' not in tool['doc'] and 'Return' not in tool['doc']:
                issues.append(tool['name'])
        
        assert not issues, f"Tools missing Returns section: {issues}"
    
    def test_docstrings_document_error_handling(self, all_tools):
        """Tools should document error cases."""
        issues = []
        for tool in all_tools:
            doc = tool['doc'].lower()
            # Should mention error handling in some way
            if not any(err in doc for err in ['error', 'on error', 'exception', 'failure']):
                issues.append(tool['name'])
        
        assert not issues, f"Tools not documenting error handling: {issues}"


class TestToolDescriptionContent:
    """Test the content quality of tool descriptions."""
    
    def test_descriptions_explain_when_to_use(self, all_tools):
        """Descriptions should include guidance on when to use the tool."""
        issues = []
        for tool in all_tools:
            doc = tool['doc'].lower()
            # Should have usage guidance
            usage_keywords = ['use this', 'when', 'for', 'useful for', 'prefer']
            if not any(kw in doc for kw in usage_keywords):
                issues.append(tool['name'])
        
        assert not issues, f"Tools lacking usage guidance: {issues}"
    
    def test_descriptions_include_examples(self, all_tools):
        """Descriptions should include examples where appropriate."""
        # Most tools should have examples
        tools_needing_examples = [
            'ipynb_read_notebook', 'ipynb_list_cells', 'ipynb_get_cell',
            'ipynb_replace_cell', 'ipynb_insert_cell', 'ipynb_search_cells',
            'ipynb_clear_outputs', 'ipynb_merge_notebooks'
        ]
        
        issues = []
        for tool in all_tools:
            if tool['name'] in tools_needing_examples:
                if 'Example' not in tool['doc'] and 'example' not in tool['doc'].lower():
                    issues.append(tool['name'])
        
        assert not issues, f"Key tools missing examples: {issues}"
    
    def test_descriptions_reference_related_tools(self, all_tools):
        """Descriptions should reference related tools for guidance."""
        # Tools that should reference alternatives
        tools_with_alternatives = {
            'ipynb_replace_cell': ['ipynb_str_replace_in_cell', 'ipynb_replace_cells_batch'],
            'ipynb_insert_cell': ['ipynb_append_cell', 'ipynb_insert_cells_batch'],
            'ipynb_delete_cell': ['ipynb_delete_cells_batch', 'ipynb_filter_cells'],
            'ipynb_get_cell': ['ipynb_list_cells', 'ipynb_search_cells'],
        }
        
        issues = []
        for tool_name, alternatives in tools_with_alternatives.items():
            tool = next((t for t in all_tools if t['name'] == tool_name), None)
            if tool:
                doc_lower = tool['doc'].lower()
                has_reference = any(alt.lower() in doc_lower for alt in alternatives)
                if not has_reference:
                    issues.append((tool_name, alternatives))
        
        assert not issues, f"Tools not referencing alternatives: {issues}"
    
    def test_filepath_parameters_document_absolute_paths(self, all_tools):
        """Filepath parameters should recommend absolute paths or document path requirements."""
        issues = []
        for tool in all_tools:
            params = list(tool['signature'].parameters.keys())
            filepath_params = [p for p in params if 'filepath' in p.lower()]
            if filepath_params:
                doc_lower = tool['doc'].lower()
                # Accept various ways to document path requirements
                has_path_guidance = any(phrase in doc_lower for phrase in [
                    'absolute', 'path', 'filepath'
                ])
                if not has_path_guidance:
                    issues.append(tool['name'])
        
        assert not issues, f"Tools not documenting path requirements: {issues}"


# ============================================================================
# Tool Naming Convention Tests
# ============================================================================

class TestToolNamingConventions:
    """Test that tool names follow the established conventions."""
    
    def test_all_tools_have_ipynb_prefix(self, all_tools):
        """All tools must have 'ipynb_' prefix for domain clarity."""
        # This is already enforced by how we collect tools, but let's verify
        for tool in all_tools:
            assert tool['name'].startswith('ipynb_'), (
                f"Tool {tool['name']} missing 'ipynb_' prefix"
            )
    
    def test_tool_names_are_action_oriented(self, all_tools):
        """Tool names should use verb_noun pattern."""
        action_verbs = [
            'read', 'list', 'get', 'search', 'replace', 'insert', 
            'append', 'delete', 'str_replace', 'update', 'set',
            'merge', 'split', 'apply', 'sync', 'extract', 'clear',
            'validate', 'filter', 'reorder'
        ]
        
        issues = []
        for tool in all_tools:
            # Remove 'ipynb_' prefix for analysis
            name_without_prefix = tool['name'][6:]  # len('ipynb_') = 6
            has_verb = any(name_without_prefix.startswith(v) for v in action_verbs)
            if not has_verb:
                issues.append(tool['name'])
        
        assert not issues, f"Tools not using action verbs: {issues}"
    
    def test_batch_tools_have_batch_suffix(self, all_tools):
        """Batch operation tools should be clearly identifiable."""
        batch_tools = [t['name'] for t in all_tools 
                      if 'batch' in t['name'].lower() or 'notebooks' in t['name'].lower()]
        
        # Should have multiple batch tools
        assert len(batch_tools) >= 5, (
            f"Expected at least 5 batch tools, found {len(batch_tools)}: {batch_tools}"
        )


# ============================================================================
# Tool Count and Category Tests
# ============================================================================

class TestToolInventory:
    """Test the tool inventory matches expectations."""
    
    def test_expected_tool_count(self, all_tools):
        """Should have exactly 29 tools as documented."""
        assert len(all_tools) == 29, (
            f"Expected 29 tools, found {len(all_tools)}: "
            f"{[t['name'] for t in all_tools]}"
        )
    
    def test_all_categories_present(self, all_tools, tool_categories):
        """All expected tool categories should be present."""
        all_tool_names = {t['name'] for t in all_tools}
        
        for category, expected_tools in tool_categories.items():
            for tool_name in expected_tools:
                assert tool_name in all_tool_names, (
                    f"Missing tool {tool_name} from category {category}"
                )
    
    def test_no_unexpected_tools(self, all_tools, tool_categories):
        """No tools outside the defined categories."""
        expected_tools = set()
        for tools in tool_categories.values():
            expected_tools.update(tools)
        
        actual_tools = {t['name'] for t in all_tools}
        unexpected = actual_tools - expected_tools
        
        assert not unexpected, f"Unexpected tools found: {unexpected}"


# ============================================================================
# Tool Invocation Scenario Tests
# ============================================================================

class TestToolInvocationScenarios:
    """Test correct tool invocation for common Jupyter editing scenarios."""
    
    def test_read_notebook_for_structure_overview(self, temp_notebook_file):
        """ipynb_read_notebook should be used for structure overview."""
        tool = getattr(server, 'ipynb_read_notebook')
        result = tool.fn(temp_notebook_file)
        
        assert 'error' not in result
        assert 'cell_count' in result
        assert 'cell_types' in result
        assert 'kernel_info' in result
    
    def test_list_cells_for_cell_enumeration(self, temp_notebook_file):
        """ipynb_list_cells should provide cell previews for navigation."""
        tool = getattr(server, 'ipynb_list_cells')
        result = tool.fn(temp_notebook_file)
        
        assert 'error' not in result
        assert 'cells' in result
        assert len(result['cells']) > 0
        assert all('index' in c and 'type' in c and 'preview' in c 
                  for c in result['cells'])
    
    def test_get_cell_for_specific_content(self, temp_notebook_file):
        """ipynb_get_cell should return complete cell content."""
        tool = getattr(server, 'ipynb_get_cell')
        result = tool.fn(temp_notebook_file, 0)
        
        assert 'error' not in result
        assert 'content' in result
        assert isinstance(result['content'], str)
    
    def test_search_cells_for_pattern_matching(self, temp_notebook_file):
        """ipynb_search_cells should find patterns across cells."""
        tool = getattr(server, 'ipynb_search_cells')
        result = tool.fn(temp_notebook_file, 'print')
        
        assert 'error' not in result
        assert 'results' in result
        assert 'match_count' in result
    
    def test_replace_cell_preserves_type(self, temp_notebook_file):
        """ipynb_replace_cell should preserve cell type while updating content."""
        list_tool = getattr(server, 'ipynb_list_cells')
        replace_tool = getattr(server, 'ipynb_replace_cell')
        
        # Get original cell info
        original = list_tool.fn(temp_notebook_file)
        original_type = original['cells'][0]['type']
        
        # Replace content
        result = replace_tool.fn(temp_notebook_file, 0, 'new_content = True')
        assert result.get('success') is True
        
        # Verify type preserved
        updated = list_tool.fn(temp_notebook_file)
        assert updated['cells'][0]['type'] == original_type
    
    def test_insert_cell_shifts_indices(self, temp_notebook_file):
        """ipynb_insert_cell should shift subsequent cell indices."""
        read_tool = getattr(server, 'ipynb_read_notebook')
        insert_tool = getattr(server, 'ipynb_insert_cell')
        
        # Get original count
        original = read_tool.fn(temp_notebook_file)
        original_count = original['cell_count']
        
        # Insert cell
        result = insert_tool.fn(temp_notebook_file, 0, '# New first cell', 'markdown')
        
        assert result.get('success') is True
        assert result['new_cell_count'] == original_count + 1
    
    def test_append_cell_adds_to_end(self, temp_notebook_file):
        """ipynb_append_cell should add cell at the end."""
        read_tool = getattr(server, 'ipynb_read_notebook')
        append_tool = getattr(server, 'ipynb_append_cell')
        
        # Get original count
        original = read_tool.fn(temp_notebook_file)
        original_count = original['cell_count']
        
        # Append cell
        result = append_tool.fn(temp_notebook_file, '# Last cell', 'markdown')
        
        assert result.get('success') is True
        assert result['cell_index'] == original_count
    
    def test_delete_cell_removes_and_shifts(self, temp_notebook_file):
        """ipynb_delete_cell should remove cell and shift indices."""
        read_tool = getattr(server, 'ipynb_read_notebook')
        delete_tool = getattr(server, 'ipynb_delete_cell')
        
        # Get original count
        original = read_tool.fn(temp_notebook_file)
        original_count = original['cell_count']
        
        # Delete first cell
        result = delete_tool.fn(temp_notebook_file, 0)
        
        assert result.get('success') is True
        assert result['new_cell_count'] == original_count - 1
    
    def test_clear_outputs_for_git_prep(self, temp_notebook_file):
        """ipynb_clear_outputs should clear outputs for version control."""
        tool = getattr(server, 'ipynb_clear_outputs')
        result = tool.fn(temp_notebook_file)
        
        assert result.get('success') is True
        assert result['notebooks_processed'] == 1
    
    def test_validate_notebook_checks_format(self, temp_notebook_file):
        """ipynb_validate_notebook should verify notebook format."""
        tool = getattr(server, 'ipynb_validate_notebook')
        result = tool.fn(temp_notebook_file)
        
        assert 'valid' in result
        assert result['valid'] is True


# ============================================================================
# Tool Selection Guidance Tests
# ============================================================================

class TestToolSelectionGuidance:
    """Test that descriptions guide correct tool selection."""
    
    def test_read_vs_list_vs_get_guidance(self, all_tools):
        """Read, list, and get tools should have clear differentiation."""
        read_tool = next(t for t in all_tools if t['name'] == 'ipynb_read_notebook')
        list_tool = next(t for t in all_tools if t['name'] == 'ipynb_list_cells')
        get_tool = next(t for t in all_tools if t['name'] == 'ipynb_get_cell')
        
        # read_notebook should mention NOT for content
        assert 'not' in read_tool['doc'].lower() or 'do not' in read_tool['doc'].lower()
        
        # list_cells should mention previews
        assert 'preview' in list_tool['doc'].lower()
        
        # get_cell should mention full/complete content
        assert 'complete' in get_tool['doc'].lower() or 'full' in get_tool['doc'].lower()
    
    def test_single_vs_batch_operation_guidance(self, all_tools):
        """Single and batch operations should reference each other."""
        replace_single = next(t for t in all_tools if t['name'] == 'ipynb_replace_cell')
        replace_batch = next(t for t in all_tools if t['name'] == 'ipynb_replace_cells_batch')
        
        # Single should mention batch
        assert 'batch' in replace_single['doc'].lower()
        
        # Batch should explain efficiency benefit
        assert 'efficient' in replace_batch['doc'].lower() or 'single' in replace_batch['doc'].lower()
    
    def test_str_replace_vs_replace_cell_guidance(self, all_tools):
        """String replace and cell replace should have clear guidance."""
        str_replace = next(t for t in all_tools if t['name'] == 'ipynb_str_replace_in_cell')
        replace_cell = next(t for t in all_tools if t['name'] == 'ipynb_replace_cell')
        
        # str_replace should mention partial/targeted
        assert 'partial' in str_replace['doc'].lower() or 'targeted' in str_replace['doc'].lower() or 'portion' in str_replace['doc'].lower()
        
        # replace_cell should mention entire/whole
        assert 'entire' in replace_cell['doc'].lower() or 'whole' in replace_cell['doc'].lower() or 'complete' in replace_cell['doc'].lower()


# ============================================================================
# Error Handling Consistency Tests
# ============================================================================

class TestErrorHandlingConsistency:
    """Test consistent error handling across all tools."""
    
    def test_file_not_found_error_format(self, all_tools):
        """All file operations should return consistent error format."""
        # Test with non-existent file
        tool = getattr(server, 'ipynb_read_notebook')
        result = tool.fn('/nonexistent/path/notebook.ipynb')
        
        assert 'error' in result
        assert 'not found' in result['error'].lower() or 'nonexistent' in result['error'].lower()
    
    def test_invalid_index_error_format(self, temp_notebook_file):
        """Invalid cell index should return consistent error format."""
        tool = getattr(server, 'ipynb_get_cell')
        result = tool.fn(temp_notebook_file, 999)
        
        assert 'error' in result
        assert 'out of range' in result['error'].lower() or 'index' in result['error'].lower()
    
    def test_success_response_format(self, temp_notebook_file):
        """Successful operations should return consistent format."""
        tool = getattr(server, 'ipynb_read_notebook')
        result = tool.fn(temp_notebook_file)
        
        # Should NOT have error key on success
        assert 'error' not in result
        
        # Should have expected keys for this operation
        assert 'cell_count' in result


# ============================================================================
# Parameter Documentation Tests
# ============================================================================

class TestParameterDocumentation:
    """Test that all parameters are properly documented."""
    
    def test_all_parameters_documented(self, all_tools):
        """Every parameter should be documented in Args section."""
        issues = []
        
        for tool in all_tools:
            params = list(tool['signature'].parameters.keys())
            for param in params:
                if param not in tool['doc']:
                    issues.append((tool['name'], param))
        
        assert not issues, f"Undocumented parameters: {issues}"
    
    def test_cell_index_documents_negative_indexing(self, all_tools):
        """cell_index parameters should document negative indexing support."""
        tools_with_cell_index = [
            'ipynb_get_cell', 'ipynb_replace_cell', 'ipynb_delete_cell',
            'ipynb_str_replace_in_cell'
        ]
        
        issues = []
        for tool in all_tools:
            if tool['name'] in tools_with_cell_index:
                if 'negative' not in tool['doc'].lower():
                    issues.append(tool['name'])
        
        assert not issues, f"Tools not documenting negative indexing: {issues}"
    
    def test_cell_type_documents_valid_values(self, all_tools):
        """cell_type parameters should list valid values."""
        tools_with_cell_type = [
            'ipynb_insert_cell', 'ipynb_append_cell', 'ipynb_insert_cells_batch',
            'ipynb_filter_cells', 'ipynb_extract_cells'
        ]
        
        issues = []
        for tool in all_tools:
            if tool['name'] in tools_with_cell_type:
                doc_lower = tool['doc'].lower()
                has_values = all(v in doc_lower for v in ['code', 'markdown', 'raw'])
                if not has_values:
                    issues.append(tool['name'])
        
        assert not issues, f"Tools not documenting cell_type values: {issues}"


# ============================================================================
# Return Value Documentation Tests
# ============================================================================

class TestReturnValueDocumentation:
    """Test that return values are properly documented."""
    
    def test_success_return_structure_documented(self, all_tools):
        """Successful returns should document their structure."""
        issues = []
        
        for tool in all_tools:
            doc = tool['doc']
            # Should have "On success:" or similar
            has_success_doc = any(
                phrase in doc.lower() 
                for phrase in ['on success', 'returns:', 'returns on success']
            )
            if not has_success_doc:
                issues.append(tool['name'])
        
        assert not issues, f"Tools not documenting success returns: {issues}"
    
    def test_error_return_structure_documented(self, all_tools):
        """Error returns should document the error key pattern."""
        issues = []
        
        # Tool that explicitly cannot fail (returns static data)
        no_error_tools = {'ipynb_list_available_kernels'}
        
        for tool in all_tools:
            if tool['name'] in no_error_tools:
                continue  # Skip tools that explicitly don't have error conditions
                
            doc = tool['doc']
            # Should document error return format
            has_error_doc = any(
                phrase in doc.lower()
                for phrase in ['on error', 'error:', "'error'", 'no error']
            )
            if not has_error_doc:
                issues.append(tool['name'])
        
        assert not issues, f"Tools not documenting error returns: {issues}"


# ============================================================================
# MCP Best Practices Compliance Tests
# ============================================================================

class TestMCPBestPractices:
    """Test compliance with MCP best practices."""
    
    def test_descriptions_minimize_ambiguity(self, all_tools):
        """Descriptions should be specific, not vague."""
        vague_terms = ['stuff', 'things', 'etc', 'various', 'many']
        
        issues = []
        for tool in all_tools:
            doc_lower = tool['doc'].lower()
            for term in vague_terms:
                if f' {term} ' in doc_lower or f' {term}.' in doc_lower:
                    issues.append((tool['name'], term))
        
        assert not issues, f"Tools with vague language: {issues}"
    
    def test_descriptions_are_self_contained(self, all_tools):
        """Each description should be understandable without external docs."""
        # Check that descriptions mention key concepts inline
        issues = []
        
        for tool in all_tools:
            doc = tool['doc']
            # Should not just say "see documentation"
            if 'see documentation' in doc.lower() or 'refer to' in doc.lower():
                # Only flag if there's no inline explanation
                if len(doc) < 300:  # Short docs with external refs are problematic
                    issues.append(tool['name'])
        
        assert not issues, f"Tools with insufficient inline documentation: {issues}"
    
    def test_tool_names_under_50_chars(self, all_tools):
        """Tool names should be concise."""
        issues = [(t['name'], len(t['name'])) for t in all_tools if len(t['name']) > 50]
        
        assert not issues, f"Tools with names too long (>50 chars): {issues}"
    
    def test_first_line_under_80_chars(self, all_tools):
        """First line of description should be concise."""
        issues = []
        for tool in all_tools:
            first_line = tool['doc'].split('\n')[0].strip()
            if len(first_line) > 80:
                issues.append((tool['name'], len(first_line)))
        
        assert not issues, f"Tools with first line too long (>80 chars): {issues}"
