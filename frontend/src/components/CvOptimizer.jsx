import React, { useState, useEffect } from 'react';
import { Sparkles, X, Copy, Check, Wand2, ArrowRight, RefreshCw, FileText } from 'lucide-react';

export default function CvOptimizer({ selectedJob, userCvText, onClose }) {
  const [experience, setExperience] = useState('');
  const [jobRequirement, setJobRequirement] = useState('');
  const [generatedBullet, setGeneratedBullet] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (selectedJob) {
      const req = selectedJob.required_skills || selectedJob.key_responsibilities || selectedJob.job_description || '';
      setJobRequirement(req);
    }
    if (userCvText) {
      setExperience(userCvText);
    }
  }, [selectedJob, userCvText]);

  const handleGenerate = async () => {
    if (!experience.trim() || !jobRequirement.trim()) {
      setError('Please provide both candidate experience and job requirement.');
      return;
    }

    setLoading(true);
    setError(null);
    setCopied(false);

    try {
      const response = await fetch('/api/generate-cv', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_experience: experience,
          job_requirement: jobRequirement
        })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to generate CV bullet point.');
      }

      setGeneratedBullet(data.generated_bullet || '');
    } catch (err) {
      setError(err.message || 'Error communicating with Generative AI model.');
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    if (generatedBullet) {
      navigator.clipboard.writeText(generatedBullet);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (!selectedJob) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 dark:bg-black/60 backdrop-blur-sm animate-fade-in">
      <div className="glass-panel w-full max-w-2xl rounded-2xl overflow-hidden border border-[var(--border-color)] shadow-2xl relative flex flex-col max-h-[90vh]">
        
        {/* Modal Header */}
        <div className="px-5 py-4 bg-[var(--footer-bg)] border-b border-[var(--border-color)] flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded bg-[var(--meta-tag-bg)] text-[var(--meta-tag-text)] border border-[var(--meta-tag-border)]">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-[var(--text-heading)]">
                FLAN-T5 CV Optimizer
              </h3>
              <p className="text-[10px] text-[var(--sub-text)]">
                Rewriting experience for <span className="text-[var(--text-color)] font-medium">{selectedJob.job_title}</span>
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-[var(--sub-text)] hover:text-[var(--text-heading)] rounded-lg hover:bg-[var(--btn-secondary-bg)] transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Content */}
        <div className="p-5 overflow-y-auto space-y-4 flex-1">
          {error && (
            <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs">
              {error}
            </div>
          )}

          {/* Inputs Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-1">
              <label className="text-[10px] font-bold text-[var(--sub-text)] uppercase tracking-wider flex items-center gap-1">
                <FileText className="w-3 h-3 text-indigo-455" />
                Your Experience / Skill Text
              </label>
              <textarea
                value={experience}
                onChange={(e) => setExperience(e.target.value)}
                rows={5}
                placeholder="Describe your achievement..."
                className="w-full glass-input p-3 rounded-xl text-xs leading-relaxed focus:ring-1 focus:ring-indigo-500 font-sans"
              />
            </div>

            <div className="space-y-1">
              <label className="text-[10px] font-bold text-[var(--sub-text)] uppercase tracking-wider flex items-center gap-1">
                <Wand2 className="w-3 h-3 text-indigo-455" />
                Target Job Requirement
              </label>
              <textarea
                value={jobRequirement}
                onChange={(e) => setJobRequirement(e.target.value)}
                rows={5}
                placeholder="Paste the required skill or role responsibility..."
                className="w-full glass-input p-3 rounded-xl text-xs leading-relaxed focus:ring-1 focus:ring-indigo-500 font-sans"
              />
            </div>
          </div>

          {/* Generate Button */}
          <div className="flex justify-center pt-2">
            <button
              onClick={handleGenerate}
              disabled={loading}
              className="px-4 py-2 rounded-xl bg-[var(--btn-secondary-bg)] hover:bg-[var(--btn-secondary-bg)]/80 text-[var(--btn-secondary-text)] border border-[var(--btn-secondary-border)] hover:text-[var(--text-heading)] font-semibold text-xs flex items-center gap-1.5 transition-colors cursor-pointer shadow-sm"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Sparkles className="w-3.5 h-3.5" />
                  Optimize Experience Bullet
                </>
              )}
            </button>
          </div>

          {/* Result */}
          {generatedBullet && (
            <div className="mt-2 p-4 rounded-xl bg-[var(--inner-card-bg)] border border-[var(--border-color)] space-y-2.5 animate-fade-in">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-500 flex items-center gap-1">
                  <Check className="w-3.5 h-3.5" />
                  AI-Optimized Bullet Point
                </span>
                <button
                  onClick={handleCopy}
                  className="px-2 py-1 rounded bg-[var(--meta-tag-bg)] hover:bg-[var(--meta-tag-bg)]/85 text-[var(--meta-tag-text)] text-[10px] font-medium flex items-center gap-1 border border-[var(--meta-tag-border)] transition-colors cursor-pointer"
                >
                  {copied ? (
                    <>
                      <Check className="w-3 h-3 text-emerald-400" />
                      Copied!
                    </>
                  ) : (
                    <>
                      <Copy className="w-3 h-3" />
                      Copy to Clipboard
                    </>
                  )}
                </button>
              </div>

              <p className="text-xs font-medium text-[var(--text-heading)] leading-relaxed bg-[var(--input-bg)] p-3 rounded-lg border border-[var(--border-color)]">
                "{generatedBullet}"
              </p>

              <div className="text-[9px] text-[var(--sub-text)] flex items-center gap-1.5">
                <ArrowRight className="w-2.5 h-2.5 text-indigo-500" />
                Prompt applied: <code className="text-indigo-400 font-mono">Rewrite the following candidate experience to align with the job requirement...</code>
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-5 py-3 bg-[var(--footer-bg)] border-t border-[var(--border-color)] flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-[var(--btn-secondary-bg)] hover:bg-[var(--btn-secondary-bg)]/85 text-[var(--btn-secondary-text)] border border-[var(--btn-secondary-border)] hover:text-[var(--text-heading)] text-xs font-medium transition-colors cursor-pointer"
          >
            Close
          </button>
        </div>

      </div>
    </div>
  );
}

