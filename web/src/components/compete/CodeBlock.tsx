interface CodeBlockProps {
  children: string;
  language?: string;
}

export function CodeBlock({ children, language = "bash" }: CodeBlockProps) {
  return (
    <div className="bg-surface-900 border border-surface-600 rounded-lg overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-surface-600">
        <span className="text-xs text-gray-500 font-mono">{language}</span>
      </div>
      <pre className="p-4 overflow-x-auto">
        <code className="text-sm font-mono text-gray-300 leading-relaxed whitespace-pre">
          {children}
        </code>
      </pre>
    </div>
  );
}
