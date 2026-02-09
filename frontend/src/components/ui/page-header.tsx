"use client";

import { LucideIcon } from "lucide-react";

interface PageHeaderProps {
  icon: LucideIcon;
  title: string;
  description?: string;
  iconColor?: string;
  iconBg?: string;
  children?: React.ReactNode;
}

export function PageHeader({ 
  icon: Icon, 
  title, 
  description, 
  iconColor = "text-indigo-600",
  iconBg = "bg-indigo-100",
  children 
}: PageHeaderProps) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4 sm:p-6 mb-6 shadow-sm">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="flex items-center gap-3 sm:gap-4 min-w-0">
          <div className={`w-10 h-10 sm:w-12 sm:h-12 ${iconBg} rounded-xl flex items-center justify-center shrink-0`}>
            <Icon className={`w-5 h-5 sm:w-6 sm:h-6 ${iconColor}`} />
          </div>
          <div className="min-w-0">
            <h1 className="text-lg sm:text-xl font-semibold text-slate-900 truncate">{title}</h1>
            {description && <p className="text-slate-500 text-xs sm:text-sm mt-0.5 line-clamp-2">{description}</p>}
          </div>
        </div>
        {children && <div className="flex items-center gap-2 shrink-0">{children}</div>}
      </div>
    </div>
  );
}
