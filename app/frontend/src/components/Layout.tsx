import { NavLink } from "react-router-dom";
import { Activity, Map, Table2, BarChart3, Moon, Sun } from "lucide-react";
import { cn } from "@/lib/utils";
import { useTheme } from "./ThemeProvider";

const NAV = [
  { to: "/", label: "Predict", icon: Activity, end: true },
  { to: "/map", label: "Map", icon: Map },
  { to: "/areas", label: "Top Areas", icon: Table2 },
  { to: "/models", label: "Models", icon: BarChart3 },
];

function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <svg viewBox="0 0 32 32" className="h-7 w-7">
        <rect width="32" height="32" rx="7" className="fill-card" />
        <path
          d="M6 22 L13 10 L19 18 L26 8"
          fill="none"
          stroke="hsl(var(--primary))"
          strokeWidth="2.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle cx="26" cy="8" r="2.6" fill="hsl(var(--primary))" />
        <circle cx="13" cy="10" r="2" fill="hsl(var(--accent))" />
      </svg>
      <div className="leading-none">
        <div className="text-sm font-bold tracking-tight text-foreground">
          Testing-Flow
        </div>
        <div className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
          Traffic Intelligence
        </div>
      </div>
    </div>
  );
}

export function Layout({ children }: { children: React.ReactNode }) {
  const { theme, toggle } = useTheme();

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-40 border-b border-border glass">
        <div className="container flex h-14 items-center justify-between">
          <Logo />

          <nav className="hidden items-center gap-1 rounded-lg border border-border bg-card/60 p-1 md:flex">
            {NAV.map(({ to, label, icon: Icon, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                className={({ isActive }) =>
                  cn(
                    "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                    isActive
                      ? "bg-primary text-primary-foreground shadow-sm"
                      : "text-muted-foreground hover:bg-secondary hover:text-foreground"
                  )
                }
              >
                <Icon className="h-4 w-4" />
                {label}
              </NavLink>
            ))}
          </nav>

          <button
            onClick={toggle}
            aria-label="Toggle theme"
            className="flex h-9 w-9 items-center justify-center rounded-md border border-border text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
          >
            {theme === "dark" ? (
              <Sun className="h-4 w-4" />
            ) : (
              <Moon className="h-4 w-4" />
            )}
          </button>
        </div>

        {/* Mobile nav */}
        <nav className="flex items-center gap-1 overflow-x-auto border-t border-border px-3 py-2 md:hidden no-scrollbar">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                cn(
                  "flex shrink-0 items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-secondary"
                )
              }
            >
              <Icon className="h-4 w-4" />
              {label}
            </NavLink>
          ))}
        </nav>
      </header>

      <main className="container py-6">{children}</main>

      <footer className="border-t border-border py-5">
        <div className="container flex flex-col items-center justify-between gap-2 text-xs text-muted-foreground sm:flex-row">
          <span>
            Testing-Flow — leakage-controlled forecasts on 8,057 Bengaluru traffic
            events (Nov 2023 – Apr 2024).
          </span>
          <span className="font-mono">closure · priority · duration · hotspot · manpower</span>
        </div>
      </footer>
    </div>
  );
}
