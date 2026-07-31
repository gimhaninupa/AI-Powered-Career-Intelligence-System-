import React from 'react';
import { Briefcase, Sparkles, Target, CheckCircle2, ChevronRight, Award } from 'lucide-react';

export default function JobResults({ jobs, onSelectOptimize, selectedJobId }) {
  if (!jobs || jobs.length === 0) {
    return null;
  }

  const getScoreColor = (percentage) => {
    if (percentage >= 75) return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30';
    if (percentage >= 50) return 'text-indigo-400 bg-indigo-500/10 border-indigo-500/30';
    return 'text-amber-400 bg-amber-500/10 border-amber-500/30';
  };

  const getProgressColor = (percentage) => {
    if (percentage >= 75) return 'bg-gradient-to-r from-emerald-500 to-teal-400';
    if (percentage >= 50) return 'bg-gradient-to-r from-indigo-500 to-purple-500';
    return 'bg-gradient-to-r from-amber-500 to-orange-400';
  };

  return (
    <section className="mt-10 space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center gap-2 font-display">
            <Target className="w-6 h-6 text-indigo-400" />
            Top 5 Semantic Job Matches
          </h2>
          <p className="text-sm text-gray-400 mt-1">
            Calculated via 384-d Sentence Transformer Embeddings & Nearest Neighbors (KNN)
          </p>
        </div>
        <span className="px-3 py-1 text-xs font-semibold rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
          {jobs.length} Matches Found
        </span>
      </div>

      <div className="grid grid-cols-1 gap-5">
        {jobs.map((job, index) => {
          const isSelected = selectedJobId === job.id;
          return (
            <div
              key={job.id || index}
              className={`glass-card p-6 rounded-2xl relative overflow-hidden transition-all duration-300 ${
                isSelected ? 'ring-2 ring-indigo-500 bg-indigo-950/20' : ''
              }`}
            >
              {/* Top Row: Rank Badge, Title, Field, Match Score */}
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-gray-700/50">
                <div className="flex items-start gap-3">
                  <div className="w-9 h-9 rounded-xl bg-gray-800 flex items-center justify-center font-bold text-gray-300 border border-gray-700 text-sm">
                    #{index + 1}
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white tracking-wide">
                      {job.job_title}
                    </h3>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-md text-xs font-medium bg-gray-800 text-indigo-300 border border-gray-700">
                        <Briefcase className="w-3 h-3" />
                        {job.job_field}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Match Percentage Badge */}
                <div className="flex items-center gap-3">
                  <div className="text-right">
                    <div className="text-xs text-gray-400 uppercase font-semibold tracking-wider">
                      Semantic Similarity
                    </div>
                    <div className="text-2xl font-extrabold text-white">
                      {job.match_percentage}%
                    </div>
                  </div>
                  <div className={`px-3 py-2 rounded-xl border flex flex-col items-center justify-center min-w-[70px] ${getScoreColor(job.match_percentage)}`}>
                    <Award className="w-5 h-5 mb-0.5" />
                    <span className="text-[10px] font-bold tracking-wider uppercase">Match</span>
                  </div>
                </div>
              </div>

              {/* Progress Bar */}
              <div className="w-full bg-gray-800 h-2 rounded-full my-4 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-1000 ${getProgressColor(job.match_percentage)}`}
                  style={{ width: `${Math.min(job.match_percentage, 100)}%` }}
                />
              </div>

              {/* Details grid: Responsibilities & Requirements */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 my-4 text-sm">
                <div className="bg-gray-900/60 p-4 rounded-xl border border-gray-800">
                  <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <ChevronRight className="w-3.5 h-3.5 text-indigo-400" />
                    Job Description & Role Summary
                  </h4>
                  <p className="text-gray-300 leading-relaxed line-clamp-3">
                    {job.job_description || job.key_responsibilities || "Key industry responsibilities."}
                  </p>
                </div>

                <div className="bg-gray-900/60 p-4 rounded-xl border border-gray-800">
                  <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                    Required Skills & Qualifications
                  </h4>
                  <p className="text-gray-300 leading-relaxed line-clamp-3">
                    {job.required_skills || "Required experience and core qualifications."}
                  </p>
                </div>
              </div>

              {/* Action Button */}
              <div className="mt-4 pt-4 border-t border-gray-800/80 flex justify-end">
                <button
                  onClick={() => onSelectOptimize(job)}
                  className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-medium text-sm flex items-center gap-2 shadow-lg shadow-indigo-600/25 hover:shadow-indigo-600/40 transition-all duration-200"
                >
                  <Sparkles className="w-4 h-4 text-purple-200" />
                  Optimize CV for this Role
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
