"use client";

import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { AlertCircle, PlayCircle, RotateCcw } from "lucide-react";

interface ResumeTestDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  groupName: string;
  existingCount: number;
  totalTests: number;
  onResume: () => void;
  onRestart: () => void;
  onCancel: () => void;
}

export function ResumeTestDialog({
  open,
  onOpenChange,
  groupName,
  existingCount,
  totalTests,
  onResume,
  onRestart,
  onCancel
}: ResumeTestDialogProps) {
  const newTests = totalTests - existingCount;
  
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertCircle className="w-5 h-5 text-amber-600" />
            Bu Grupta Daha Önce Test Yapılmış
          </DialogTitle>
          <DialogDescription className="text-base pt-4">
            <div className="space-y-3">
              <p>
                <span className="font-semibold text-slate-900">"{groupName}"</span> grubunda 
                <span className="font-semibold text-indigo-600"> {existingCount} kayıtlı sonuç</span> bulundu.
              </p>
              
              <div className="bg-slate-50 rounded-lg p-4 space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-slate-600">Toplam test sayısı:</span>
                  <span className="font-semibold text-slate-900">{totalTests}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-600">Yapılmış testler:</span>
                  <span className="font-semibold text-emerald-600">{existingCount}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-600">Kalan testler:</span>
                  <span className="font-semibold text-indigo-600">{newTests}</span>
                </div>
              </div>
              
              <p className="text-sm text-slate-600">
                Ne yapmak istersiniz?
              </p>
            </div>
          </DialogDescription>
        </DialogHeader>
        
        <DialogFooter className="flex-col sm:flex-col gap-2 sm:gap-2">
          <Button 
            onClick={onResume}
            className="w-full bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700"
          >
            <PlayCircle className="w-4 h-4 mr-2" />
            Devam Et ({newTests} yeni test)
          </Button>
          
          <Button 
            onClick={onRestart}
            variant="outline"
            className="w-full border-amber-300 text-amber-700 hover:bg-amber-50"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Baştan Başla (Tüm sonuçları sil)
          </Button>
          
          <Button 
            onClick={onCancel}
            variant="ghost"
            className="w-full"
          >
            İptal
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}