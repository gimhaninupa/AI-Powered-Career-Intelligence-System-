import React from 'react';
import { Briefcase, Sparkles, Target, ChevronRight } from 'lucide-react';

export default function JobResults({ jobs, onSelectOptimize, selectedJobId }) {
  if (!jobs || jobs.length === 0) {
    return null;
  }

  const getScoreColor = (percentage) => {
    if (percentage >= 75) return 'text-emerald-500 dark:text-emerald-400 border-emerald-500/20 dark:border-emerald-950 bg-emerald-500/10 dark:bg-emerald-950/20';
    if (percentage >= 50) return 'text-indigo-600 dark:text-indigo-400 border-indigo-500/20 dark:border-indigo-950 bg-indigo-500/10 dark:bg-indigo-950/20';
    return 'text-amber-600 dark:text-amber-400 border-amber-500/20 dark:border-amber-950 bg-amber-500/10 dark:bg-amber-950/20';
  };

  return (
    <section className="space-y-4 animate-fade-in">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-bold text-[var(--sub-text)] uppercase tracking-wider flex items-center gap-2">
          <Target className="w-4 h-4 text-indigo-500" />
          Semantic Job Matches
        </h3>
        <span className="text-[10px] text-[var(--sub-text)] font-mono">
          KNN Score Sorted
        </span>
      </div>

      <div className="space-y-3">
        {jobs.map((job, index) => {
          const isSelected = selectedJobId === job.id;
          return (
            <div
              key={job.id || index}
              className={`glass-card p-5 rounded-xl border border-[var(--border-color)] transition-all duration-200 ${
                isSelected ? 'ring-1 ring-indigo-500/50 bg-indigo-500/5 dark:bg-indigo-950/10' : ''
              }`}
            >
              {/* Header */}
              <div className="flex items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <span className="text-xs font-mono text-[var(--sub-text)]">
                    0{index + 1}
                  </span>
                  <div>
                    <h4 className="text-sm font-semibold">
                      {job.job_title}
                    </h4>
                    <div className="flex items-center gap-2 mt-0.5">
                      <span className="inline-flex items-center gap-1 text-[10px] text-[var(--sub-text)] font-medium">
                        <Briefcase className="w-2.5 h-2.5" />
                        {job.job_field}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Score */}
                <div className={`px-2.5 py-0.5 text-xs font-semibold rounded-full border ${getScoreColor(job.match_percentage)}`}>
                  {job.match_percentage}% Match
                </div>
              </div>

              {/* Description Snippet */}
              <div className="mt-3 text-xs text-[var(--sub-text)] leading-relaxed border-t border-[var(--border-color)] pt-2.5">
                <p className="line-clamp-2">
                  {job.job_description || job.key_responsibilities || "No description details provided."}
                </p>
              </div>

              {/* Action */}
              <div className="mt-3 flex items-center justify-between text-xs pt-2 border-t border-[var(--border-color)]">
                <div className="text-[10px] text-[var(--sub-text)]">
                  Required skills: <span className="text-[var(--text-color)] opacity-80">{job.required_skills ? job.required_skills.split(',').slice(0, 3).join(', ') : 'N/A'}</span>
                </div>
                <button
                  onClick={() => onSelectOptimize(job)}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[var(--btn-secondary-bg)] text-[var(--btn-secondary-text)] font-medium hover:text-[var(--text-heading)] transition-colors cursor-pointer border border-[var(--btn-secondary-border)]"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  Optimize CV
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

