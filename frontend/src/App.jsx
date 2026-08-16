import React, { useState, useEffect } from 'react';
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
  AlertCircle,
  Upload,
  Sun,
  Moon
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
  const [uploading, setUploading] = useState(false);
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('theme') || 'dark';
  });

  useEffect(() => {
    const root = window.document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload-cv', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to parse the file.');
      }

      setCvText(data.extracted_text);
    } catch (err) {
      setError(err.message || 'Error uploading and parsing file.');
    } finally {
      setUploading(false);
      event.target.value = null;
    }
  };

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
    <div className="min-h-screen flex flex-col bg-[var(--bg-color)]">
      {/* Header */}
      <header className="border-b border-[var(--border-color)] bg-[var(--header-bg)] backdrop-blur-md sticky top-0 z-45">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-600/80 flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-semibold tracking-tight">
                AI Career Intelligence
              </h1>
              <p className="text-[10px] text-gray-500">
                SentenceTransformers (MiniLM-L6-v2) + SpaCy + FLAN-T5
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={toggleTheme}
              className="p-1.5 rounded-lg border border-[var(--border-color)] bg-[var(--dark-indicator)] text-[var(--sub-text)] hover:text-[var(--text-heading)] transition-colors cursor-pointer"
              title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            >
              {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
            <div className="flex items-center gap-2 text-xs text-[var(--sub-text)] bg-[var(--dark-indicator)] px-2.5 py-1 rounded-md border border-[var(--border-color)]">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
              System Active
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* Left Column: CV Upload & Inputs */}
          <div className="lg:col-span-5 space-y-6">
            <div className="space-y-1">
              <h2 className="text-lg font-bold tracking-tight text-[var(--text-heading)]">Resume Ingestion</h2>
              <p className="text-xs text-[var(--sub-text)]">Upload your CV file or paste technical objectives below.</p>
            </div>

            <div className="glass-panel p-6 rounded-2xl border border-[var(--border-color)] space-y-5">
              
              {/* File Upload Zone */}
              <div className="border border-dashed border-[var(--border-color)] hover:border-indigo-500/50 bg-[var(--inner-card-bg)] rounded-xl p-6 text-center transition-colors relative group">
                <input
                  type="file"
                  accept=".pdf,.docx,.doc,.txt"
                  onChange={handleFileUpload}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                  disabled={uploading}
                />
                <div className="flex flex-col items-center justify-center gap-2">
                  {uploading ? (
                    <>
                      <RefreshCw className="w-6 h-6 text-indigo-400 animate-spin" />
                      <p className="text-xs text-[var(--sub-text)]">Extracting text...</p>
                    </>
                  ) : (
                    <>
                      <Upload className="w-6 h-6 text-indigo-400 group-hover:scale-105 transition-transform" />
                      <p className="text-xs font-medium text-[var(--text-color)]">
                        Drag & drop CV or <span className="text-indigo-400">browse</span>
                      </p>
                      <p className="text-[10px] text-[var(--sub-text)]">PDF, DOCX, or TXT</p>
                    </>
                  )}
                </div>
              </div>

              {/* Text Area */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-[11px] font-bold text-[var(--sub-text)] uppercase tracking-wider">
                    Raw CV Content
                  </label>
                  
                  {/* Preset Pills */}
                  <div className="flex items-center gap-1.5">
                    {SAMPLE_CVS.map((sample, idx) => (
                      <button
                        key={idx}
                        onClick={() => handleLoadSample(sample.text)}
                        className="px-2 py-0.5 rounded bg-[var(--btn-secondary-bg)] hover:bg-[var(--btn-secondary-bg)]/80 text-[var(--btn-secondary-text)] text-[10px] border border-[var(--btn-secondary-border)] transition-colors cursor-pointer"
                      >
                        {sample.label.split(' ')[0]}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="relative">
                  <textarea
                    value={cvText}
                    onChange={(e) => setCvText(e.target.value)}
                    rows={8}
                    placeholder="Or paste your CV details here..."
                    className="w-full glass-input p-3 rounded-xl text-sm leading-relaxed resize-none focus:ring-1 focus:ring-indigo-500 font-sans"
                  />
                  {cvText && (
                    <span className="absolute bottom-2 right-2 text-[10px] text-[var(--sub-text)] font-mono">
                      {cvText.length} chars
                    </span>
                  )}
                </div>
              </div>

              {error && (
                <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  {error}
                </div>
              )}

              {/* Action Button */}
              <button
                onClick={handleMatchJobs}
                disabled={loading || !cvText.trim()}
                className="w-full py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-semibold text-sm flex items-center justify-center gap-2 transition-colors shadow-sm cursor-pointer"
              >
                {loading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Analyzing Embeddings & KNN...
                  </>
                ) : (
                  <>
                    <Search className="w-4 h-4" />
                    Analyze & Find Match
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Right Column: AI Analytics & Recommendations */}
          <div className="lg:col-span-7 space-y-6">
            {!results ? (
              <div className="border border-[var(--border-color)] bg-[var(--inner-card-bg)] rounded-2xl p-12 text-center h-full flex flex-col items-center justify-center min-h-[400px]">
                <Cpu className="w-8 h-8 text-[var(--sub-text)] mb-3" />
                <h3 className="text-sm font-semibold text-[var(--text-heading)]">No Analytics Yet</h3>
                <p className="text-xs text-[var(--sub-text)] max-w-sm mt-1">
                  Upload a resume or select a sample on the left, then run the analyzer to see matching job profiles.
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Predictions & NLP tags */}
                <div className="glass-panel p-5 rounded-2xl border border-[var(--border-color)] space-y-4">
                  <div className="flex items-center justify-between border-b border-[var(--border-color)] pb-3">
                    <span className="text-[11px] font-bold text-[var(--sub-text)] uppercase tracking-wider">
                      Intelligence Insights
                    </span>
                    {results.predicted_field && (
                      <span className="text-xs font-semibold text-[var(--meta-tag-text)] bg-[var(--meta-tag-bg)] border border-[var(--meta-tag-border)] px-2.5 py-0.5 rounded-full">
                        Field: {results.predicted_field}
                      </span>
                    )}
                  </div>

                  <div className="space-y-2">
                    <span className="text-xs font-medium text-[var(--sub-text)] block">Extracted Skill Keywords:</span>
                    <div className="flex flex-wrap gap-1.5">
                      {results.extracted_skills && results.extracted_skills.length > 0 ? (
                        results.extracted_skills.map((skill, i) => (
                          <span
                            key={i}
                            className="px-2.5 py-0.5 rounded bg-[var(--btn-secondary-bg)] text-[var(--btn-secondary-text)] text-xs border border-[var(--btn-secondary-border)]"
                          >
                            {skill}
                          </span>
                        ))
                      ) : (
                        <span className="text-xs text-[var(--sub-text)] italic">No skills detected.</span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Job Matches Component */}
                {results.top_matches && (
                  <JobResults
                    jobs={results.top_matches}
                    onSelectOptimize={(job) => setSelectedJobForOptimization(job)}
                    selectedJobId={selectedJobForOptimization?.id}
                  />
                )}
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-[var(--border-color)] bg-[var(--footer-bg)] py-6 mt-12 text-center text-[10px] text-[var(--sub-text)]">
        <p>AI Career Intelligence &copy; 2026. Powered by SentenceTransformers (MiniLM-L6-v2) + SpaCy + FLAN-T5.</p>
      </footer>

      {/* CV Optimizer Modal */}
      {selectedJobForOptimization && (
        <CvOptimizer
          selectedJob={selectedJobForOptimization}
          userCvText={cvText}
          onClose={() => setSelectedJobForOptimization(null)}
        />
      )}
    </div>
  );
}

