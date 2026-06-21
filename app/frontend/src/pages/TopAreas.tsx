import { useEffect, useMemo, useState } from "react";
import {
  type ColumnDef,
  type SortingState,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getSortedRowModel,
  useReactTable,
} from "@tanstack/react-table";
import {
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Search,
  Flame,
} from "lucide-react";
import { api } from "@/lib/api";
import type { AreaRow } from "@/lib/types";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { InfoTip } from "@/components/ui/popover";
import { LoadingState, ErrorState } from "@/components/ui/feedback";
import { cn } from "@/lib/utils";
import { LEVEL_STYLES, minutesToHuman, pct, num, titleCase, riskLevel } from "@/lib/format";

function RiskBar({ score }: { score: number }) {
  const level = riskLevel(score);
  const style = LEVEL_STYLES[level];
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-secondary">
        <div
          className={cn("h-full rounded-full", style.dot)}
          style={{ width: `${Math.min(score * 1.6, 100)}%` }}
        />
      </div>
      <span className={cn("w-9 text-right text-xs font-semibold tabular-nums", style.text)}>
        {score.toFixed(0)}
      </span>
    </div>
  );
}

export default function TopAreas() {
  const [data, setData] = useState<AreaRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [sorting, setSorting] = useState<SortingState>([
    { id: "n_events", desc: true },
  ]);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    api.areas().then(setData).catch((e) => setError((e as Error).message));
  }, []);

  const columns = useMemo<ColumnDef<AreaRow>[]>(
    () => [
      {
        accessorKey: "area",
        header: "Area (police station)",
        cell: ({ row }) => (
          <div className="flex items-center gap-2">
            <span className="font-medium text-foreground">{row.original.area}</span>
            {row.original.chronic_count > 20 && (
              <Flame className="h-3.5 w-3.5 text-danger" />
            )}
          </div>
        ),
        enableSorting: true,
      },
      {
        accessorKey: "n_events",
        header: "Events",
        cell: ({ getValue }) => (
          <span className="tabular-nums">{(getValue() as number).toLocaleString()}</span>
        ),
      },
      {
        accessorKey: "risk_score",
        header: "Risk",
        cell: ({ getValue }) => <RiskBar score={getValue() as number} />,
      },
      {
        accessorKey: "closure_rate",
        header: "Closure rate",
        cell: ({ getValue }) => (
          <span className="tabular-nums">{pct(getValue() as number)}</span>
        ),
      },
      {
        accessorKey: "high_priority_rate",
        header: "High priority",
        cell: ({ getValue }) => (
          <span className="tabular-nums">{pct(getValue() as number)}</span>
        ),
      },
      {
        accessorKey: "avg_duration_min",
        header: "Avg clearance",
        cell: ({ getValue }) => (
          <span className="tabular-nums">{minutesToHuman(getValue() as number)}</span>
        ),
        sortUndefined: "last",
      },
      {
        accessorKey: "chronic_count",
        header: "Chronic cells",
        cell: ({ getValue }) => (
          <span className="tabular-nums">{getValue() as number}</span>
        ),
      },
      {
        accessorKey: "avg_officers",
        header: "Avg officers",
        cell: ({ getValue }) => (
          <span className="tabular-nums">{num(getValue() as number, 1)}</span>
        ),
      },
      {
        accessorKey: "top_causes",
        header: "Top causes",
        enableSorting: false,
        cell: ({ getValue }) => (
          <div className="flex flex-wrap gap-1">
            {(getValue() as string[]).slice(0, 2).map((c) => (
              <Badge key={c} variant="secondary" className="text-[10px]">
                {titleCase(c)}
              </Badge>
            ))}
          </div>
        ),
      },
    ],
    []
  );

  const table = useReactTable({
    data,
    columns,
    state: { sorting, globalFilter: filter },
    onSortingChange: setSorting,
    onGlobalFilterChange: setFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    globalFilterFn: (row, _id, value) =>
      row.original.area.toLowerCase().includes(String(value).toLowerCase()),
  });

  if (error) return <ErrorState message={error} />;
  if (!data.length) return <LoadingState label="Loading areas…" />;

  return (
    <div>
      <div className="mb-4 flex flex-col justify-between gap-3 sm:flex-row sm:items-end">
        <div>
          <h1 className="flex items-center gap-2 text-xl font-bold tracking-tight">
            Top areas by risk
            <InfoTip title="Risk score">
              A 0–100 blend of predicted closure rate (45%), average chronic-hotspot
              risk (35%) and predicted high-priority rate (20%) across every event
              in the area. Click any column header to sort.
            </InfoTip>
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {data.length} police-station jurisdictions · sortable on every column.
          </p>
        </div>
        <div className="relative w-full sm:w-64">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Filter areas…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="pl-9"
          />
        </div>
      </div>

      <Card className="overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              {table.getHeaderGroups().map((hg) => (
                <tr key={hg.id} className="border-b border-border bg-card">
                  {hg.headers.map((header) => {
                    const sortable = header.column.getCanSort();
                    const sorted = header.column.getIsSorted();
                    return (
                      <th
                        key={header.id}
                        className="whitespace-nowrap px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground"
                      >
                        {sortable ? (
                          <button
                            onClick={header.column.getToggleSortingHandler()}
                            className="group inline-flex items-center gap-1.5 transition-colors hover:text-foreground"
                          >
                            {flexRender(
                              header.column.columnDef.header,
                              header.getContext()
                            )}
                            {sorted === "asc" ? (
                              <ArrowUp className="h-3.5 w-3.5 text-primary" />
                            ) : sorted === "desc" ? (
                              <ArrowDown className="h-3.5 w-3.5 text-primary" />
                            ) : (
                              <ArrowUpDown className="h-3.5 w-3.5 opacity-40 group-hover:opacity-70" />
                            )}
                          </button>
                        ) : (
                          flexRender(
                            header.column.columnDef.header,
                            header.getContext()
                          )
                        )}
                      </th>
                    );
                  })}
                </tr>
              ))}
            </thead>
            <tbody>
              {table.getRowModel().rows.map((row, i) => (
                <tr
                  key={row.id}
                  className={cn(
                    "border-b border-border/60 transition-colors hover:bg-secondary/40",
                    i % 2 === 1 && "bg-secondary/10"
                  )}
                >
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id} className="whitespace-nowrap px-4 py-2.5">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {table.getRowModel().rows.length === 0 && (
        <p className="py-8 text-center text-sm text-muted-foreground">
          No areas match “{filter}”.
        </p>
      )}
    </div>
  );
}
