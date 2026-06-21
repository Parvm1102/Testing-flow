import { Loader2, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

export function Spinner({ className }: { className?: string }) {
  return <Loader2 className={cn("h-4 w-4 animate-spin", className)} />;
}

export function LoadingState({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex h-64 flex-col items-center justify-center gap-3 text-muted-foreground">
      <Spinner className="h-6 w-6 text-primary" />
      <p className="text-sm">{label}</p>
    </div>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="flex h-64 flex-col items-center justify-center gap-3 text-center text-muted-foreground">
      <AlertTriangle className="h-6 w-6 text-warning" />
      <p className="max-w-md text-sm">{message}</p>
    </div>
  );
}
