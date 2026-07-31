import React, { useState } from 'react';
import { 
  Brain, 
  Sparkles, 
  Search, 
  FileText, 
  Cpu, 
  CheckCircle, 
  Zap, 
  Layers, 
  RefreshCw,
  AlertCircle
} from 'lucide-react';
import JobResults from './components/JobResults';
import CvOptimizer from './components/CvOptimizer';

const SAMPLE_CVS = [
  {
    label: "Big Data & Python Engineer",
    text: "Big data analytics working and database warehouse manager with robust experience in handling all kinds of data. I have also used multiple cloud infrastructure services and am well acquainted with them. Currently in search of role that offers more of development. Technical skills: Python, PySpark, Hadoop, Hive, AWS, SQL, Data Pipeline Architecture."
  },
  {
    label: "Retail Bank Teller & Customer Sales",
    text: "Front-line retail banking customer service professional with 3+ years experience processing daily financial transactions, cash vault balancing, loan product cross-selling, and loan application assistance. Strong cash handling, accuracy, and customer satisfaction record."
  },
  {
    label: "Full-Stack Software Engineer",
    text: "Software Developer experienced in React, Node.js, Python Flask API development, PostgreSQL, and cloud deployments. Passionate about machine learning integration, clean code architecture, and high throughput microservices."
  }
];

export default function App() {
  const [cvText, setCvText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [selectedJobForOptimization, setSelectedJobForOptimization] = useState(null);

  const handleMatchJobs = async () => {
    if (!cvText.trim()) {
      setError('Please enter or paste your CV / Career Objective text first.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/match-jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cv_text: cvText })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to match CV to job descriptions.');
      }

      setResults(data);
    } catch (err) {
      setError(err.message || 'Server error occurred while matching jobs.');
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSample = (sampleText) => {
    setCvText(sampleText);
    setError(null);
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-gray-800 bg-gray-950/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/30">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-extrabold text-white tracking-tight flex items-center gap-2 font-display">
                AI Career Intelligence
                <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 font-mono">
                  v1.0 ML
                </span>
              </h1>
              <p className="text-[11px] text-gray-400 hidden sm:block">
                SentenceTransformers (MiniLM-L6-v2) + SpaCy NER + FLAN-T5
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-xs text-gray-400 bg-gray-900 px-3 py-1.5 rounded-lg border border-gray-800">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              Backend API Connected
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        
        {/* Hero Banner */}
        <section className="text-center space-y-3 py-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-xs font-semibold uppercase tracking-wider">
            <Zap className="w-3.5 h-3.5 text-indigo-400" />
            Mathematical Semantic Matching Engine
          </div>
          <h2 className="text-3xl sm:text-5xl font-extrabold text-white tracking-tight font-display max-w-3xl mx-auto leading-tight">
            Match Your CV to Target Roles & <span className="bg-gradient-to-r from-indigo-400 via-purple-400 to-emerald-400 bg-clip-text text-transparent">Optimize with Generative AI</span>
          </h2>
          <p className="text-gray-400 text-sm sm:text-base max-w-2xl mx-auto">
            Paste your unstructured raw CV or career objective. Our vector embedding model computes Cosine Similarity across 600+ job descriptions and rewrites your experience bullets using FLAN-T5.
          </p>
        </section>

        {/* Input Card */}
        <section className="glass-panel p-6 sm:p-8 rounded-3xl space-y-6 shadow-2xl relative">
          
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <label className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <FileText className="w-4 h-4 text-indigo-400" />
              Paste Raw Resume / Career Objective
            </label>

            {/* Quick Sample Presets */}
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs text-gray-400 font-medium">Try Sample:</span>
              {SAMPLE_CVS.map((sample, idx) => (
                <button
                  key={idx}
                  onClick={() => handleLoadSample(sample.text)}
                  className="px-2.5 py-1 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-300 text-xs font-medium border border-gray-700 transition-colors"
                >
                  {sample.label}
                </button>
              ))}
            </div>
          </div>

          <div className="relative">
            <textarea
              value={cvText}
              onChange={(e) => setCvText(e.target.value)}
              rows={6}
              placeholder="Paste your CV text, skills, past positions, or career objective here..."
              className="w-full glass-input p-4 rounded-2xl text-base leading-relaxed resize-y focus:ring-2 focus:ring-indigo-500 font-sans"
            />
            {cvText && (
              <span className="absolute bottom-3 right-3 text-xs text-gray-500 font-mono">
                {cvText.length} characters
              </span>
            )}
          </div>

          {error && (
            <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-300 text-sm flex items-center gap-2">
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              {error}
            </div>
          )}

          {/* Submit Action */}
          <div className="flex items-center justify-between pt-2">
            <button
              onClick={handleMatchJobs}
              disabled={loading || !cvText.trim()}
              className="w-full sm:w-auto px-8 py-4 rounded-2xl bg-gradient-to-r from-indigo-600 via-purple-600 to-indigo-700 hover:from-indigo-500 hover:to-purple-500 disabled:opacity-50 text-white font-bold text-base flex items-center justify-center gap-3 shadow-xl shadow-indigo-600/30 transition-all duration-200"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-5 h-5 animate-spin text-indigo-200" />
                  Processing Vector Embeddings & KNN...
                </>
              ) : (
                <>
                  <Search className="w-5 h-5 text-indigo-200" />
                  Analyze CV & Find Jobs
                </>
              )}
            </button>

            {results && (
              <span className="hidden sm:flex items-center gap-1.5 text-xs text-emerald-400 font-medium">
                <CheckCircle className="w-4 h-4" />
                Analysis Complete
              </span>
            )}
          </div>
        </section>

        {/* Extracted Skills Badges (if available) */}
        {results && (
          <section className="glass-card p-6 rounded-2xl space-y-4 animate-fade-in">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Cpu className="w-4 h-4 text-purple-400" />
                SpaCy Extracted Candidate Skills & Entities
              </h3>
              {results.predicted_field && (
                <span className="text-xs text-indigo-300 bg-indigo-900/50 px-3 py-1 rounded-lg border border-indigo-700">
                  Predicted Field: <strong>{results.predicted_field}</strong>
                </span>
              )}
            </div>

            <div className="flex flex-wrap gap-2">
              {results.extracted_skills && results.extracted_skills.length > 0 ? (
                results.extracted_skills.map((skill, i) => (
                  <span
                    key={i}
                    className="px-3 py-1 rounded-xl bg-gray-800 text-indigo-200 text-xs font-semibold border border-indigo-500/20 shadow-sm"
                  >
                    {skill}
                  </span>
                ))
              ) : (
                <span className="text-xs text-gray-500 italic">No explicit noun-chunk skills detected.</span>
              )}
            </div>
          </section>
        )}

        {/* Results Component */}
        {results && results.top_matches && (
          <JobResults
            jobs={results.top_matches}
            onSelectOptimize={(job) => setSelectedJobForOptimization(job)}
            selectedJobId={selectedJobForOptimization?.id}
          />
        )}

        {/* Generative AI CV Optimizer Modal */}
        {selectedJobForOptimization && (
          <CvOptimizer
            selectedJob={selectedJobForOptimization}
            userCvText={cvText}
            onClose={() => setSelectedJobForOptimization(null)}
          />
        )}

      </main>

      {/* Footer */}
      <footer className="border-t border-gray-900 bg-gray-950 py-6 mt-16 text-center text-xs text-gray-500 space-y-1">
        <p>AI-Powered Career Intelligence System &copy; 2026. Built with Python, Flask, SentenceTransformers, SpaCy, FLAN-T5 & React.</p>
      </footer>
    </div>
  );
}
