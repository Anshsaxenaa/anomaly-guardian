import { useState } from "react";
import { Check, Copy } from "lucide-react";
import { cn } from "@/lib/utils";

interface CodeBlockProps {
  code: string;
  language?: string;
  filename?: string;
  className?: string;
}

export function CodeBlock({ code, language = "plaintext", filename, className }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={cn("relative group rounded-lg border border-border overflow-hidden", className)}>
      {filename && (
        <div className="flex items-center justify-between px-4 py-2 bg-secondary/50 border-b border-border">
          <span className="text-xs font-mono text-muted-foreground">{filename}</span>
          <span className="text-xs font-mono text-primary/70">{language}</span>
        </div>
      )}
      <div className="relative">
        <pre className="p-4 overflow-x-auto scrollbar-thin bg-secondary/30">
          <code className="text-sm font-mono text-foreground/90 leading-relaxed">{code}</code>
        </pre>
        <button
          onClick={handleCopy}
          className="absolute top-3 right-3 p-2 rounded-md bg-secondary/80 hover:bg-secondary border border-border opacity-0 group-hover:opacity-100 transition-opacity"
          aria-label="Copy code"
        >
          {copied ? (
            <Check className="w-4 h-4 text-success" />
          ) : (
            <Copy className="w-4 h-4 text-muted-foreground" />
          )}
        </button>
      </div>
    </div>
  );
}
