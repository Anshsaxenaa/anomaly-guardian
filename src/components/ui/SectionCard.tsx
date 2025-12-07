import { cn } from "@/lib/utils";
import { ReactNode } from "react";

interface SectionCardProps {
  title: string;
  subtitle?: string;
  icon?: ReactNode;
  children: ReactNode;
  className?: string;
  badge?: string;
  badgeVariant?: "primary" | "success" | "warning" | "destructive";
}

export function SectionCard({ 
  title, 
  subtitle, 
  icon, 
  children, 
  className,
  badge,
  badgeVariant = "primary"
}: SectionCardProps) {
  const badgeColors = {
    primary: "bg-primary/20 text-primary",
    success: "bg-success/20 text-success",
    warning: "bg-warning/20 text-warning",
    destructive: "bg-destructive/20 text-destructive",
  };

  return (
    <section className={cn("section-card animate-fade-in", className)}>
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          {icon && (
            <div className="p-2 rounded-lg bg-primary/10 text-primary">
              {icon}
            </div>
          )}
          <div>
            <h2 className="text-xl font-semibold text-foreground">{title}</h2>
            {subtitle && <p className="text-sm text-muted-foreground mt-0.5">{subtitle}</p>}
          </div>
        </div>
        {badge && (
          <span className={cn("metric-badge", badgeColors[badgeVariant])}>
            {badge}
          </span>
        )}
      </div>
      {children}
    </section>
  );
}
