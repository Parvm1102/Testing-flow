import * as React from "react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Check, ChevronsUpDown, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { titleCase } from "@/lib/format";

interface SearchSelectProps {
  options: string[];
  value?: string;
  onChange: (value: string | undefined) => void;
  placeholder?: string;
  clearable?: boolean;
  pretty?: boolean; // title-case the display label
}

/** Searchable single-select for high-cardinality fields (police station, junction). */
export function SearchSelect({
  options,
  value,
  onChange,
  placeholder = "Select…",
  clearable = true,
  pretty = false,
}: SearchSelectProps) {
  const [open, setOpen] = React.useState(false);
  const [query, setQuery] = React.useState("");

  const filtered = React.useMemo(() => {
    const q = query.trim().toLowerCase();
    const list = q
      ? options.filter((o) => o.toLowerCase().includes(q))
      : options;
    return list.slice(0, 200);
  }, [options, query]);

  const label = value ? (pretty ? titleCase(value) : value) : placeholder;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          role="combobox"
          aria-expanded={open}
          className={cn(
            "flex h-9 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus:outline-none focus:ring-2 focus:ring-ring",
            !value && "text-muted-foreground"
          )}
        >
          <span className="line-clamp-1 text-left">{label}</span>
          <span className="flex items-center gap-1">
            {clearable && value && (
              <X
                className="h-3.5 w-3.5 opacity-60 hover:opacity-100"
                onClick={(e) => {
                  e.stopPropagation();
                  onChange(undefined);
                }}
              />
            )}
            <ChevronsUpDown className="h-4 w-4 opacity-60" />
          </span>
        </button>
      </PopoverTrigger>
      <PopoverContent className="w-[var(--radix-popover-trigger-width)] p-0" align="start">
        <div className="border-b border-border p-2">
          <input
            autoFocus
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Type to search…"
            className="h-8 w-full rounded-md bg-secondary px-2.5 text-sm text-foreground outline-none placeholder:text-muted-foreground"
          />
        </div>
        <div className="max-h-60 overflow-y-auto p-1 no-scrollbar">
          {filtered.length === 0 && (
            <p className="px-2 py-4 text-center text-xs text-muted-foreground">
              No matches
            </p>
          )}
          {filtered.map((opt) => (
            <button
              key={opt}
              type="button"
              onClick={() => {
                onChange(opt);
                setOpen(false);
                setQuery("");
              }}
              className="flex w-full items-center justify-between rounded-sm px-2 py-1.5 text-left text-sm hover:bg-secondary"
            >
              <span className="line-clamp-1">{pretty ? titleCase(opt) : opt}</span>
              {value === opt && <Check className="h-4 w-4 text-primary" />}
            </button>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  );
}
