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
      // Prefill requirement from job details
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
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fade-in">
      <div className="glass-panel w-full max-w-3xl rounded-3xl overflow-hidden border border-indigo-500/30 shadow-2xl relative flex flex-col max-h-[90vh]">
        
        {/* Modal Header */}
        <div className="px-6 py-5 bg-gray-900/90 border-b border-gray-800 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-purple-500/20 text-purple-400 border border-purple-500/30">
              <Sparkles className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-white">
                FLAN-T5 CV Bullet Generator
              </h3>
              <p className="text-xs text-gray-400">
                Aligning your raw experience with <span className="text-indigo-300 font-semibold">{selectedJob.job_title}</span>
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-white rounded-xl hover:bg-gray-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Content - Scrollable */}
        <div className="p-6 overflow-y-auto space-y-5 flex-1">

          {error && (
            <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-300 text-sm">
              {error}
            </div>
          )}

          {/* Inputs Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-gray-300 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <FileText className="w-3.5 h-3.5 text-indigo-400" />
                Candidate Experience / Resume Text
              </label>
              <textarea
                value={experience}
                onChange={(e) => setExperience(e.target.value)}
                rows={4}
                placeholder="Paste specific achievement or experience bullet point..."
                className="w-full glass-input p-3.5 rounded-xl text-sm leading-relaxed focus:ring-2 focus:ring-indigo-500"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-gray-300 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Wand2 className="w-3.5 h-3.5 text-purple-400" />
                Target Job Requirement
              </label>
              <textarea
                value={jobRequirement}
                onChange={(e) => setJobRequirement(e.target.value)}
                rows={4}
                placeholder="Target skills or required qualifications..."
                className="w-full glass-input p-3.5 rounded-xl text-sm leading-relaxed focus:ring-2 focus:ring-indigo-500"
              />
            </div>
          </div>

          {/* Generate Trigger Button */}
          <div className="flex justify-center pt-2">
            <button
              onClick={handleGenerate}
              disabled={loading}
              className="px-6 py-3 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 disabled:opacity-50 text-white font-semibold text-sm flex items-center gap-2 shadow-lg shadow-purple-600/30 transition-all duration-200"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin text-purple-200" />
                  Generating with google/flan-t5-base...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4 text-purple-200" />
                  Rewrite & Optimize Bullet Point
                </>
              )}
            </button>
          </div>

          {/* Generated Result Container */}
          {generatedBullet && (
            <div className="mt-4 p-5 rounded-2xl bg-gradient-to-br from-indigo-950/60 to-purple-950/40 border border-indigo-500/30 space-y-3 animate-fade-in">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-emerald-400 flex items-center gap-1.5">
                  <Check className="w-4 h-4" />
                  AI-Optimized Bullet Point
                </span>
                <button
                  onClick={handleCopy}
                  className="px-3 py-1.5 rounded-lg bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-200 text-xs font-medium flex items-center gap-1.5 border border-indigo-500/30 transition-colors"
                >
                  {copied ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                      Copied!
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      Copy to Clipboard
                    </>
                  )}
                </button>
              </div>

              <p className="text-base font-medium text-white leading-relaxed bg-black/40 p-4 rounded-xl border border-white/5">
                "{generatedBullet}"
              </p>

              <div className="text-[11px] text-gray-400 flex items-center gap-2 pt-1">
                <ArrowRight className="w-3 h-3 text-indigo-400" />
                Prompt applied: <code className="text-indigo-300 font-mono">Rewrite the following candidate experience to align with the job requirement...</code>
              </div>
            </div>
          )}

        </div>

        {/* Modal Footer */}
        <div className="px-6 py-4 bg-gray-900/90 border-t border-gray-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl bg-gray-800 hover:bg-gray-700 text-gray-300 text-sm font-medium transition-colors"
          >
            Close
          </button>
        </div>

      </div>
    </div>
  );
}
