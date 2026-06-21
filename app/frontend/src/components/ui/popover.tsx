import * as React from "react";
import * as PopoverPrimitive from "@radix-ui/react-popover";
import { Info } from "lucide-react";
import { cn } from "@/lib/utils";

const Popover = PopoverPrimitive.Root;
const PopoverTrigger = PopoverPrimitive.Trigger;

const PopoverContent = React.forwardRef<
  React.ElementRef<typeof PopoverPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof PopoverPrimitive.Content>
>(({ className, align = "center", sideOffset = 6, ...props }, ref) => (
  <PopoverPrimitive.Portal>
    <PopoverPrimitive.Content
      ref={ref}
      align={align}
      sideOffset={sideOffset}
      className={cn(
        "z-50 w-72 rounded-lg border border-border bg-popover p-4 text-popover-foreground shadow-lg outline-none data-[state=open]:animate-scale-in",
        className
      )}
      {...props}
    />
  </PopoverPrimitive.Portal>
));
PopoverContent.displayName = "PopoverContent";

/** Small (i) icon that reveals an explanation popover. */
export function InfoTip({
  title,
  children,
  className,
}: {
  title?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label="More information"
          className={cn(
            "inline-flex h-4 w-4 items-center justify-center rounded-full text-muted-foreground transition-colors hover:text-accent focus:outline-none focus-visible:ring-2 focus-visible:ring-ring",
            className
          )}
        >
          <Info className="h-3.5 w-3.5" />
        </button>
      </PopoverTrigger>
      <PopoverContent>
        {title && (
          <p className="mb-1.5 text-xs font-semibold text-foreground">{title}</p>
        )}
        <div className="text-xs leading-relaxed text-muted-foreground">
          {children}
        </div>
      </PopoverContent>
    </Popover>
  );
}

export { Popover, PopoverTrigger, PopoverContent };
