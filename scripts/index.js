import React, { useState, useEffect } from 'react';
import { Send, Moon, Sun, CheckCircle, AlertCircle, Loader2, TrendingUp, BarChart3 } from 'lucide-react';

const SentimentAnalysisUI = () => {
  const [darkMode, setDarkMode] = useState(false);
  const [title, setTitle] = useState('');
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [apiHealth, setApiHealth] = useState(null);

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/health');
      const data = await response.json();
      setApiHealth(data);
    } catch (err) {
      setApiHealth({ status: 'offline', model_loaded: false });
    }
  };

  const handleSubmit = async () => {
    if (!title || apiHealth?.status !== 'healthy') return;
    
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          title: title,
          text: text || null,
        }),
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleSubmit();
    }
  };

  const getSentimentColor = (sentiment) => {
    const s = sentiment?.toLowerCase();
    if (s?.includes('positive')) return darkMode ? 'text-emerald-400' : 'text-emerald-600';
    if (s?.includes('negative')) return darkMode ? 'text-red-400' : 'text-red-600';
    return darkMode ? 'text-amber-400' : 'text-amber-600';
  };

  const getSentimentBg = (sentiment) => {
    const s = sentiment?.toLowerCase();
    if (s?.includes('positive')) return darkMode ? 'bg-emerald-500/10' : 'bg-emerald-50';
    if (s?.includes('negative')) return darkMode ? 'bg-red-500/10' : 'bg-red-50';
    return darkMode ? 'bg-amber-500/10' : 'bg-amber-50';
  };

  return (
    <div className={`min-h-screen transition-colors duration-300 ${
      darkMode 
        ? 'bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900' 
        : 'bg-gradient-to-br from-slate-50 via-white to-slate-100'
    }`}>
      <header className={`border-b transition-colors ${
        darkMode ? 'border-slate-700/50 bg-slate-900/50' : 'border-slate-200 bg-white/50'
      } backdrop-blur-md sticky top-0 z-50`}>
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={`p-2 rounded-xl ${
                darkMode ? 'bg-blue-500/20' : 'bg-blue-50'
              }`}>
                <BarChart3 className={`w-6 h-6 ${
                  darkMode ? 'text-blue-400' : 'text-blue-600'
                }`} />
              </div>
              <div>
                <h1 className={`text-xl sm:text-2xl font-semibold ${
                  darkMode ? 'text-white' : 'text-slate-900'
                }`}>
                  Sentiment Analysis
                </h1>
                <p className={`text-sm ${
                  darkMode ? 'text-slate-400' : 'text-slate-600'
                }`}>
                  AI-powered sentiment detection
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <div className={`hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium ${
                apiHealth?.status === 'healthy'
                  ? darkMode ? 'bg-emerald-500/20 text-emerald-400' : 'bg-emerald-50 text-emerald-700'
                  : darkMode ? 'bg-red-500/20 text-red-400' : 'bg-red-50 text-red-700'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  apiHealth?.status === 'healthy' ? 'bg-emerald-500' : 'bg-red-500'
                } animate-pulse`} />
                {apiHealth?.status === 'healthy' ? 'API Online' : 'API Offline'}
              </div>

              <button
                onClick={() => setDarkMode(!darkMode)}
                className={`p-2.5 rounded-xl transition-all ${
                  darkMode 
                    ? 'bg-slate-800 hover:bg-slate-700 text-slate-300' 
                    : 'bg-slate-100 hover:bg-slate-200 text-slate-700'
                }`}
                aria-label="Toggle dark mode"
              >
                {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-12">
        <div className={`rounded-2xl shadow-xl backdrop-blur-md border transition-all ${
          darkMode 
            ? 'bg-slate-800/40 border-slate-700/50' 
            : 'bg-white/60 border-slate-200/50'
        }`}>
          <div className="p-6 sm:p-8">
            <div className="space-y-6">
              <div>
                <label 
                  htmlFor="title" 
                  className={`block text-sm font-medium mb-2 ${
                    darkMode ? 'text-slate-200' : 'text-slate-700'
                  }`}
                >
                  Title <span className="text-red-500">*</span>
                </label>
                <input
                  id="title"
                  type="text"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Enter product title or review headline..."
                  className={`w-full px-4 py-3 rounded-xl border transition-all focus:outline-none focus:ring-2 ${
                    darkMode
                      ? 'bg-slate-900/50 border-slate-700 text-white placeholder-slate-500 focus:ring-blue-500/50'
                      : 'bg-white border-slate-300 text-slate-900 placeholder-slate-400 focus:ring-blue-500'
                  }`}
                />
              </div>

              <div>
                <label 
                  htmlFor="text" 
                  className={`block text-sm font-medium mb-2 ${
                    darkMode ? 'text-slate-200' : 'text-slate-700'
                  }`}
                >
                  Review Text <span className={`text-xs font-normal ${
                    darkMode ? 'text-slate-400' : 'text-slate-500'
                  }`}>(Optional)</span>
                </label>
                <textarea
                  id="text"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Enter the full review text for more accurate analysis..."
                  rows={5}
                  className={`w-full px-4 py-3 rounded-xl border transition-all focus:outline-none focus:ring-2 resize-none ${
                    darkMode
                      ? 'bg-slate-900/50 border-slate-700 text-white placeholder-slate-500 focus:ring-blue-500/50'
                      : 'bg-white border-slate-300 text-slate-900 placeholder-slate-400 focus:ring-blue-500'
                  }`}
                />
              </div>

              <button
                onClick={handleSubmit}
                disabled={loading || !title || apiHealth?.status !== 'healthy'}
                className={`w-full py-3.5 px-6 rounded-xl font-medium transition-all flex items-center justify-center gap-2 ${
                  loading || !title || apiHealth?.status !== 'healthy'
                    ? darkMode 
                      ? 'bg-slate-700 text-slate-500 cursor-not-allowed' 
                      : 'bg-slate-200 text-slate-400 cursor-not-allowed'
                    : darkMode
                      ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-lg shadow-blue-500/25'
                      : 'bg-blue-600 hover:bg-blue-700 text-white shadow-lg shadow-blue-500/20'
                }`}
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Send className="w-5 h-5" />
                    Analyze Sentiment
                  </>
                )}
              </button>

              <p className={`text-xs text-center ${
                darkMode ? 'text-slate-500' : 'text-slate-400'
              }`}>
                Press Ctrl+Enter or Cmd+Enter to submit
              </p>
            </div>
          </div>
        </div>

        {(result || error) && (
          <div className="mt-6">
            {error && (
              <div className={`rounded-2xl p-6 backdrop-blur-md border ${
                darkMode
                  ? 'bg-red-500/10 border-red-500/30'
                  : 'bg-red-50 border-red-200'
              }`}>
                <div className="flex items-start gap-3">
                  <AlertCircle className={`w-5 h-5 flex-shrink-0 mt-0.5 ${
                    darkMode ? 'text-red-400' : 'text-red-600'
                  }`} />
                  <div>
                    <h3 className={`font-semibold mb-1 ${
                      darkMode ? 'text-red-300' : 'text-red-900'
                    }`}>
                      Analysis Failed
                    </h3>
                    <p className={darkMode ? 'text-red-300/80' : 'text-red-700'}>
                      {error}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {result && (
              <div className={`rounded-2xl shadow-xl backdrop-blur-md border transition-all ${
                darkMode 
                  ? 'bg-slate-800/40 border-slate-700/50' 
                  : 'bg-white/60 border-slate-200/50'
              }`}>
                <div className="p-6 sm:p-8">
                  <div className="flex items-center gap-2 mb-6">
                    <CheckCircle className={`w-5 h-5 ${
                      darkMode ? 'text-emerald-400' : 'text-emerald-600'
                    }`} />
                    <h2 className={`text-lg font-semibold ${
                      darkMode ? 'text-white' : 'text-slate-900'
                    }`}>
                      Analysis Complete
                    </h2>
                  </div>

                  <div className={`inline-flex items-center gap-2 px-6 py-3 rounded-xl mb-6 ${
                    getSentimentBg(result.prediction)
                  }`}>
                    <TrendingUp className={`w-5 h-5 ${getSentimentColor(result.prediction)}`} />
                    <span className={`text-lg font-semibold ${getSentimentColor(result.prediction)}`}>
                      {result.prediction}
                    </span>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <p className={`text-sm font-medium mb-2 ${
                        darkMode ? 'text-slate-400' : 'text-slate-600'
                      }`}>
                        Title:
                      </p>
                      <p className={`text-base ${
                        darkMode ? 'text-slate-200' : 'text-slate-800'
                      }`}>
                        {result.title}
                      </p>
                    </div>

                    {result.text && (
                      <div>
                        <p className={`text-sm font-medium mb-2 ${
                          darkMode ? 'text-slate-400' : 'text-slate-600'
                        }`}>
                          Review Text:
                        </p>
                        <p className={`text-base leading-relaxed ${
                          darkMode ? 'text-slate-300' : 'text-slate-700'
                        }`}>
                          {result.text}
                        </p>
                      </div>
                    )}

                    {result.confidence && (
                      <div>
                        <p className={`text-sm font-medium mb-2 ${
                          darkMode ? 'text-slate-400' : 'text-slate-600'
                        }`}>
                          Confidence:
                        </p>
                        <div className="flex items-center gap-3">
                          <div className={`flex-1 h-2 rounded-full overflow-hidden ${
                            darkMode ? 'bg-slate-700' : 'bg-slate-200'
                          }`}>
                            <div 
                              className={`h-full rounded-full transition-all duration-500 ${
                                darkMode ? 'bg-blue-500' : 'bg-blue-600'
                              }`}
                              style={{ width: `${result.confidence * 100}%` }}
                            />
                          </div>
                          <span className={`text-sm font-medium ${
                            darkMode ? 'text-slate-300' : 'text-slate-700'
                          }`}>
                            {(result.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        <div className={`mt-8 text-center text-sm ${
          darkMode ? 'text-slate-400' : 'text-slate-600'
        }`}>
          <p>Powered by Machine Learning • Enterprise-grade Analysis</p>
        </div>
      </main>
    </div>
  );
};

export default SentimentAnalysisUI;