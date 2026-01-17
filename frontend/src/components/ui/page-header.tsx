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
    <div className="bg-white rounded-xl border border-slate-200 p-6 mb-6 shadow-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className={`w-12 h-12 ${iconBg} rounded-xl flex items-center justify-center`}>
            <Icon className={`w-6 h-6 ${iconColor}`} />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-slate-900">{title}</h1>
            {description && <p className="text-slate-500 text-sm mt-0.5">{description}</p>}
          </div>
        </div>
        {children && <div className="flex items-center gap-2">{children}</div>}
      </div>
    </div>
  );
}
