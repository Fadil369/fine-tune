#!/usr/bin/env python3
"""
Enhanced LLM CLI with MCP Integration
Combines LM Studio local models with enterprise features
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Callable
import requests
from datetime import datetime

try:
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.theme import Theme
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Install 'rich' for enhanced UI: pip3 install rich")

# Configuration
LMS_API_URL = "http://localhost:1234/v1"
CONFIG_DIR = Path.home() / ".lmstudio"
PLUGINS_DIR = CONFIG_DIR / "plugins"
LOGS_DIR = CONFIG_DIR / "audit-logs"

# Enhanced theme
if RICH_AVAILABLE:
    theme = Theme({
        "primary": "#2b6cb8",
        "accent": "#0ea5e9",
        "success": "#10b981",
        "error": "#ef4444",
        "warning": "#ea580c"
    })
    console = Console(theme=theme)
else:
    console = None


class AuditLogger:
    """Simple audit logger for tracking CLI operations"""
    
    def __init__(self, log_dir: Path = LOGS_DIR):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    async def log_event(self, event_type: str, metadata: Optional[Dict] = None):
        """Log an event to daily audit file"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "metadata": metadata or {}
        }
        
        log_file = self.log_dir / f"audit_{datetime.now().date()}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
    async def log_error(self, error: Exception, context: Optional[Dict] = None):
        """Log errors for debugging"""
        await self.log_event(
            event_type="ERROR",
            metadata={
                "error": str(error),
                "type": type(error).__name__,
                "context": context or {}
            }
        )


class LMStudioClient:
    """Enhanced LM Studio API client with tool support"""
    
    def __init__(self, base_url: str = LMS_API_URL, audit_logger: Optional[AuditLogger] = None):
        self.base_url = base_url
        self.session = requests.Session()
        self.audit_logger = audit_logger or AuditLogger()
        
    async def chat(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Send chat request to LM Studio"""
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            # Log usage
            if "usage" in data:
                await self.audit_logger.log_event(
                    "LLM_REQUEST",
                    metadata={
                        "model": data.get("model", "unknown"),
                        "tokens": data["usage"]
                    }
                )
            
            return content
            
        except Exception as e:
            await self.audit_logger.log_error(e, {"endpoint": "chat"})
            return f"Error: {str(e)}\n\nMake sure LM Studio server is running: llm-start"
            
    async def chat_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        tool_executor: Callable,
        max_iterations: int = 10
    ) -> str:
        """Chat with tool execution support"""
        
        current_messages = messages.copy()
        
        for iteration in range(max_iterations):
            # Add tools context to system message
            tools_desc = self._format_tools_for_prompt(tools)
            
            # Create enhanced messages with tool context
            enhanced_messages = [
                {
                    "role": "system",
                    "content": f"""You are a helpful AI assistant with access to tools.

Available tools:
{tools_desc}

To use a tool, respond with JSON in this format:
{{"tool": "tool_name", "arguments": {{"param": "value"}}}}

Only use tools when necessary. For general questions, respond normally."""
                }
            ] + current_messages
            
            response = await self.chat(enhanced_messages)
            
            # Check if response contains tool call
            if self._is_tool_call(response):
                tool_call = self._parse_tool_call(response)
                
                # Execute tool
                tool_result = await tool_executor(
                    tool_name=tool_call["tool"],
                    arguments=tool_call["arguments"]
                )
                
                # Add to conversation
                current_messages.append({"role": "assistant", "content": response})
                current_messages.append({
                    "role": "user",
                    "content": f"Tool result: {json.dumps(tool_result)}"
                })
            else:
                return response
                
        return "Max iterations reached. Please try again."
        
    def _format_tools_for_prompt(self, tools: List[Dict]) -> str:
        """Format tools as markdown for prompt"""
        lines = []
        for tool in tools:
            lines.append(f"- **{tool['name']}**: {tool['description']}")
            if "parameters" in tool or "inputSchema" in tool:
                schema = tool.get("parameters", tool.get("inputSchema", {}))
                if "properties" in schema:
                    params = ", ".join(schema["properties"].keys())
                    lines.append(f"  Parameters: {params}")
        return "\n".join(lines)
        
    def _is_tool_call(self, response: str) -> bool:
        """Check if response contains tool call"""
        response_clean = response.strip()
        return response_clean.startswith("{") and '"tool"' in response_clean
        
    def _parse_tool_call(self, response: str) -> Dict:
        """Parse tool call from response"""
        try:
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            json_str = response[start:end]
            return json.loads(json_str)
        except Exception:
            return {"tool": "unknown", "arguments": {}}


class MCPManager:
    """Simplified MCP manager for local integration"""
    
    def __init__(self, config_path: Path, audit_logger: AuditLogger):
        self.config_path = config_path
        self.audit_logger = audit_logger
        self.servers = {}
        self.tools = []
        
    async def load_mcp_config(self):
        """Load MCP configuration from mcp.json"""
        mcp_file = self.config_path / "mcp.json"
        if not mcp_file.exists():
            return
            
        with open(mcp_file) as f:
            config = json.load(f)
            
        # Store server info
        self.servers = config.get("mcpServers", {})
        
        # Build tool list from server descriptions
        for name, server_config in self.servers.items():
            desc = server_config.get("description", "")
            self.tools.append({
                "name": f"mcp_{name}",
                "description": desc,
                "server": name
            })
            
    async def get_all_tools(self) -> List[Dict]:
        """Get list of available MCP tools"""
        return self.tools


class ToolRegistry:
    """Plugin/tool registry with auto-discovery"""
    
    def __init__(self, plugins_dir: Path, audit_logger: AuditLogger):
        self.plugins_dir = plugins_dir
        self.audit_logger = audit_logger
        self.tools = []
        self.plugin_instances = {}
        
    async def discover_plugins(self):
        """Discover and load Python plugins"""
        if not self.plugins_dir.exists():
            self.plugins_dir.mkdir(parents=True)
            return
        
        import sys
        import importlib.util
        
        # Add plugins dir to path
        if str(self.plugins_dir) not in sys.path:
            sys.path.insert(0, str(self.plugins_dir))
            
        # Scan for .py files
        for plugin_file in self.plugins_dir.glob("*.py"):
            if plugin_file.name.startswith('_'):
                continue
                
            try:
                # Import the module
                spec = importlib.util.spec_from_file_location(plugin_file.stem, plugin_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Look for tool classes
                    for item_name in dir(module):
                        item = getattr(module, item_name)
                        if (isinstance(item, type) and 
                            hasattr(item, 'name') and 
                            hasattr(item, 'description') and
                            hasattr(item, 'execute')):
                            
                            # Instantiate the tool
                            tool_instance = item()
                            self.plugin_instances[item.name] = tool_instance
                            
                            self.tools.append({
                                "name": item.name,
                                "description": item.description,
                                "parameters": getattr(item, 'parameters', {}),
                                "path": str(plugin_file)
                            })
            except Exception as e:
                await self.audit_logger.log_error(e, {"plugin": str(plugin_file)})
                
    async def get_tools(self) -> List[Dict]:
        """Get list of available tools"""
        return self.tools
    
    async def execute_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """Execute a plugin tool"""
        if tool_name not in self.plugin_instances:
            return {"error": f"Tool not found: {tool_name}", "success": False}
        
        try:
            tool_instance = self.plugin_instances[tool_name]
            result = await tool_instance.execute(arguments)
            return result
        except Exception as e:
            await self.audit_logger.log_error(e, {"tool": tool_name})
            return {"error": str(e), "success": False}


class EnhancedCLI:
    """Enhanced CLI with MCP and tool support"""
    
    def __init__(self, config_dir: Path = CONFIG_DIR):
        self.config_dir = config_dir
        self.audit_logger = AuditLogger()
        self.llm = LMStudioClient(audit_logger=self.audit_logger)
        self.mcp_manager = MCPManager(config_dir, self.audit_logger)
        self.tool_registry = ToolRegistry(PLUGINS_DIR, self.audit_logger)
        self.conversation_history = []
        
    async def initialize(self):
        """Initialize CLI components"""
        if console:
            console.print("[cyan]🧠 Initializing Enhanced LM Studio CLI...[/cyan]")
        else:
            print("🧠 Initializing Enhanced LM Studio CLI...")
            
        await self.audit_logger.log_event("CLI_INIT")
        
        # Load MCP config
        await self.mcp_manager.load_mcp_config()
        
        # Discover plugins
        await self.tool_registry.discover_plugins()
        
        mcp_count = len(self.mcp_manager.servers)
        tools_count = len(self.tool_registry.tools)
        
        if console:
            console.print(f"[green]✓[/green] Loaded {mcp_count} MCP servers")
            console.print(f"[green]✓[/green] Found {tools_count} plugins")
        else:
            print(f"✓ Loaded {mcp_count} MCP servers")
            print(f"✓ Found {tools_count} plugins")
            
    async def chat_loop(self, use_tools: bool = False):
        """Interactive chat session"""
        if console:
            console.print("\n[bold cyan]Enhanced LM Studio CLI[/bold cyan]")
            console.print("[dim]Type 'exit' to quit, '/tools' to list tools[/dim]\n")
        else:
            print("\nEnhanced LM Studio CLI")
            print("Type 'exit' to quit, '/tools' to list tools\n")
            
        while True:
            try:
                if console:
                    user_input = Prompt.ask("[bold blue]You[/bold blue]")
                else:
                    user_input = input("You: ")
                    
                if user_input.lower() in ["exit", "quit"]:
                    break
                    
                if user_input == "/tools":
                    await self.show_tools()
                    continue
                    
                # Log interaction
                await self.audit_logger.log_event(
                    "USER_PROMPT",
                    metadata={"length": len(user_input)}
                )
                
                # Process query
                if use_tools:
                    response = await self.process_with_tools(user_input)
                else:
                    self.conversation_history.append({"role": "user", "content": user_input})
                    response = await self.llm.chat(self.conversation_history)
                    self.conversation_history.append({"role": "assistant", "content": response})
                
                # Display response
                if console:
                    console.print("\n[bold green]Assistant[/bold green]")
                    console.print(Markdown(response))
                    console.print()
                else:
                    print(f"\nAssistant: {response}\n")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                await self.audit_logger.log_error(e)
                if console:
                    console.print(f"[red]Error: {str(e)}[/red]")
                else:
                    print(f"Error: {str(e)}")
                    
    async def process_with_tools(self, query: str) -> str:
        """
        Process query with intelligent natural language to plugin mapping.
        Analyzes intent and calls appropriate plugins directly.
        """
        # Get available tools
        mcp_tools = await self.mcp_manager.get_all_tools()
        local_tools = await self.tool_registry.get_tools()
        
        # Add to history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Analyze query to determine which plugin to use
        plugin_call = await self._analyze_intent(query)
        
        if plugin_call:
            # Execute the plugin directly
            if console:
                console.print(f"[dim]→ Using {plugin_call['tool']} plugin...[/dim]")
            else:
                print(f"→ Using {plugin_call['tool']} plugin...")
            
            result = await self._execute_tool(plugin_call['tool'], plugin_call['arguments'])
            
            # Format the result nicely
            response = self._format_result(result, plugin_call['tool'])
        else:
            # No plugin match - use regular chat
            response = await self.llm.chat(self.conversation_history)
        
        self.conversation_history.append({"role": "assistant", "content": response})
        return response
    
    async def _analyze_intent(self, query: str) -> Optional[Dict]:
        """
        Analyze user query using pattern matching to determine plugin and parameters.
        Returns plugin call dict or None if no match.
        """
        query_lower = query.lower()
        
        # Project Scout patterns
        if any(kw in query_lower for kw in ['list project', 'show project', 'my project', 'find project', 'all project']):
            action = 'list_projects'
            base_dir = '/Users/fadil369/Projects'
            
            if 'python' in query_lower:
                return {'tool': 'project_scout', 'arguments': {'action': 'find_by_language', 'language': 'python', 'base_dir': base_dir}}
            elif any(js in query_lower for js in ['javascript', 'js', 'node']):
                return {'tool': 'project_scout', 'arguments': {'action': 'find_by_language', 'language': 'javascript', 'base_dir': base_dir}}
            elif 'swift' in query_lower:
                return {'tool': 'project_scout', 'arguments': {'action': 'find_by_language', 'language': 'swift', 'base_dir': base_dir}}
            elif 'recent' in query_lower:
                return {'tool': 'project_scout', 'arguments': {'action': 'recent_activity', 'base_dir': base_dir}}
            
            return {'tool': 'project_scout', 'arguments': {'action': action, 'base_dir': base_dir}}
        
        # Git Operations patterns
        if any(kw in query_lower for kw in ['git', 'branch', 'commit', 'repository', 'repo']):
            operation = 'status'
            repo_path = '.'
            
            if 'status' in query_lower:
                operation = 'status'
            elif 'log' in query_lower or 'history' in query_lower:
                operation = 'recent_commits'
            elif 'branch' in query_lower:
                operation = 'branches'
            elif 'diff' in query_lower:
                operation = 'diff'
            elif 'stat' in query_lower:
                operation = 'stats'
            
            return {'tool': 'git_operations', 'arguments': {'operation': operation, 'repo_path': repo_path}}
        
        # System Info patterns
        if any(kw in query_lower for kw in ['system', 'cpu', 'memory', 'disk', 'process', 'resource', 'performance']):
            metric = 'overview'
            
            if 'cpu' in query_lower:
                metric = 'cpu'
            elif any(mem in query_lower for mem in ['memory', 'ram']):
                metric = 'memory'
            elif any(disk in query_lower for disk in ['disk', 'storage', 'space']):
                metric = 'disk'
            elif 'process' in query_lower:
                metric = 'processes'
            
            return {'tool': 'system_info', 'arguments': {'metric': metric}}
        
        # File Operations patterns
        if any(kw in query_lower for kw in ['search file', 'find file', 'search for', 'find .', '*.py', '*.js']):
            operation = 'search'
            path = '/Users/fadil369/Projects'
            pattern = '*'
            
            # Detect file patterns
            if '*.py' in query_lower or 'python file' in query_lower:
                pattern = '*.py'
            elif '*.js' in query_lower or 'javascript file' in query_lower:
                pattern = '*.js'
            elif '*.ts' in query_lower:
                pattern = '*.ts'
            elif '*.swift' in query_lower:
                pattern = '*.swift'
            
            return {'tool': 'file_operations', 'arguments': {'operation': operation, 'path': path, 'pattern': pattern}}
        
        # LM Studio Manager patterns
        if any(kw in query_lower for kw in ['lm studio', 'lmstudio', 'server running', 'models loaded']):
            action = 'status'
            
            if any(kw in query_lower for kw in ['running', 'status', 'is']):
                action = 'status'
            elif 'loaded' in query_lower:
                action = 'loaded_models'
            elif 'list model' in query_lower:
                action = 'list_models'
            
            return {'tool': 'lmstudio_manager', 'arguments': {'action': action}}
        
        # No plugin match
        return None
    
    def _format_result(self, result: Dict, tool_name: str) -> str:
        """Format plugin result into readable markdown text"""
        if not result.get('success', True):
            return f"❌ **Error**: {result.get('error', 'Unknown error')}"
        
        # Format based on tool type
        if tool_name == 'project_scout':
            if 'total_projects' in result:
                lines = [f"📁 **Found {result['total_projects']} projects** in `{result.get('base_directory', '~/Projects')}`:\n"]
                for p in result.get('projects', [])[:10]:
                    size = f"{p.get('size_mb', 0):.1f}MB"
                    ptype = p.get('type', 'Unknown')
                    lines.append(f"  • **{p['name']}** ({ptype}, {size})")
                if result['total_projects'] > 10:
                    lines.append(f"\n  _(and {result['total_projects'] - 10} more...)_")
                return '\n'.join(lines)
        
        elif tool_name == 'git_operations':
            if 'branch' in result:
                lines = ["🔀 **Git Status**:"]
                lines.append(f"  **Branch**: `{result['branch']}`")
                if result.get('modified'):
                    lines.append(f"  Modified: **{len(result['modified'])}** files")
                    for f in result['modified'][:5]:
                        lines.append(f"    - {f}")
                if result.get('untracked'):
                    lines.append(f"  Untracked: **{len(result['untracked'])}** files")
                lines.append(f"  **Clean**: {'✓ Yes' if result.get('clean') else '✗ No'}")
                return '\n'.join(lines)
            elif 'commits' in result:
                lines = [f"📜 **Recent Commits** ({result.get('count', 0)}):\n"]
                for c in result['commits'][:5]:
                    lines.append(f"  • `{c['hash']}` - {c['message']}")
                    lines.append(f"    _{c['author']}, {c['date']}_")
                return '\n'.join(lines)
        
        elif tool_name == 'system_info':
            if 'cpu_count' in result:
                lines = ["💻 **System Overview**:"]
                lines.append(f"  **OS**: {result.get('os')} {result.get('os_version')}")
                lines.append(f"  **CPU**: {result.get('cpu_count')} cores @ **{result.get('cpu_percent')}%** usage")
                lines.append(f"  **Memory**: **{result.get('memory_percent')}%** used")
                lines.append(f"  **Disk**: **{result.get('disk_percent')}%** used")
                return '\n'.join(lines)
            elif 'server_running' in result:
                if result['server_running']:
                    lines = ["✅ **LM Studio Status**:"]
                    lines.append(f"  Server: **Running** ✓")
                    lines.append(f"  URL: `{result.get('server_url')}`")
                    return '\n'.join(lines)
                else:
                    return "❌ LM Studio server is **not running**\n\n💡 _Start with: `llm-start`_"
        
        elif tool_name == 'file_operations':
            if 'total_matches' in result:
                lines = [f"🔍 **Found {result['total_matches']} files** matching `{result.get('pattern')}` in `{result.get('search_path')}`:\n"]
                for m in result.get('matches', [])[:10]:
                    lines.append(f"  • `{m['name']}` ({m['size_kb']:.1f} KB)")
                if result['total_matches'] > 10:
                    lines.append(f"\n  _(and {result['total_matches'] - 10} more...)_")
                return '\n'.join(lines)
        
        elif tool_name == 'lmstudio_manager':
            if 'server_running' in result:
                if result['server_running']:
                    return f"✅ LM Studio is **running** on `{result.get('server_url')}`"
                else:
                    return "❌ LM Studio is **not running**\n\n💡 _Start with: `llm-start`_"
            elif 'loaded_models' in result:
                lines = [f"📦 **Loaded Models** ({result.get('count', 0)}):\n"]
                for m in result['loaded_models']:
                    lines.append(f"  • `{m.get('model')}` ({m.get('status')}) - {m.get('size')}")
                return '\n'.join(lines)
        
        # Fallback
        return str(result)
        return response
        
    async def _execute_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """Execute a tool"""
        await self.audit_logger.log_event(
            "TOOL_EXECUTION",
            metadata={"tool": tool_name, "args_count": len(arguments)}
        )
        
        try:
            # Check if it's a local plugin
            if tool_name in self.tool_registry.plugin_instances:
                result = await self.tool_registry.execute_tool(tool_name, arguments)
                return result
            
            # Check if it's an MCP tool
            if tool_name.startswith("mcp_"):
                return {
                    "result": f"MCP tool {tool_name} would be executed here",
                    "note": "MCP execution requires active server connection",
                    "arguments": arguments
                }
            
            return {"error": f"Tool {tool_name} not found", "success": False}
        except Exception as e:
            await self.audit_logger.log_error(e, {"tool": tool_name})
            return {"error": str(e), "success": False}
        
    async def show_tools(self):
        """Display available tools"""
        mcp_tools = await self.mcp_manager.get_all_tools()
        local_tools = await self.tool_registry.get_tools()
        
        if console:
            table = Table(title="Available Tools", border_style="accent")
            table.add_column("Type", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Description", style="dim")
            
            for tool in mcp_tools:
                table.add_row("MCP", tool["name"], tool["description"])
            for tool in local_tools:
                table.add_row("Plugin", tool["name"], tool["description"])
                
            console.print(table)
        else:
            print("\nAvailable Tools:")
            print("\nMCP Servers:")
            for tool in mcp_tools:
                print(f"  - {tool['name']}: {tool['description']}")
            print("\nLocal Plugins:")
            for tool in local_tools:
                print(f"  - {tool['name']}: {tool['description']}")
            print()


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced LM Studio CLI")
    parser.add_argument("--tools", action="store_true", help="Enable tool execution")
    parser.add_argument("--config", type=Path, default=CONFIG_DIR, help="Config directory")
    
    args = parser.parse_args()
    
    cli = EnhancedCLI(config_dir=args.config)
    await cli.initialize()
    await cli.chat_loop(use_tools=args.tools)


if __name__ == "__main__":
    asyncio.run(main())
