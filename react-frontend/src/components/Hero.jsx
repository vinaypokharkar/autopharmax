import React, { useState, useEffect } from 'react';
import { ChevronRight, ArrowLeft } from 'lucide-react';

export default function AutoPharmaX() {
  const [page, setPage] = useState('home');
  // State for prediction page (can be removed if only home is needed)
  const [cellLines, setCellLines] = useState([]);
  const [drugs, setDrugs] = useState([]);
  const [selectedCellLine, setSelectedCellLine] = useState('');
  const [selectedDrug, setSelectedDrug] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  // Dummy fetch function if backend is not available
  useEffect(() => {
    const fetchOptions = async () => {
      // This is a dummy implementation. Replace with your actual API call.
      try {
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 500));
        const data = {
          cell_lines: ['MCF7', 'A549', 'HELA', 'PC3', 'HT29'],
          drugs: ['Paclitaxel', 'Doxorubicin', 'Cisplatin', 'Gemcitabine', 'Tamoxifen']
        };
        setCellLines(data.cell_lines || []);
        setDrugs(data.drugs || []);
        if (data.cell_lines?.length) setSelectedCellLine(data.cell_lines[0]);
        if (data.drugs?.length) setSelectedDrug(data.drugs[0]);
      } catch (err) {
        setError('Failed to load options');
      }
    };
    if (page === 'predict') {
        fetchOptions();
    }
  }, [page]);


  const handlePredict = async () => {
    if (!selectedCellLine || !selectedDrug) return;

    setLoading(true);
    setError('');
    // This is a dummy implementation. Replace with your actual API call.
    try {
        // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500));
      const dummyResult = {
        predicted_ln_ic50: Math.random() * 5,
        predicted_ic50: Math.exp(Math.random() * 5),
        actual_ln_ic50: Math.random() * 5,
        actual_ic50: Math.exp(Math.random() * 5),
        absolute_error: Math.random() * 0.5,
        model_name: "XGBoost_v1.2",
        num_features: 1024,
        training_date: "2023-10-26"
      };
      setResult(dummyResult);
    } catch (err) {
      setError('Prediction failed: ' + err.message);
    }
    setLoading(false);
  };


  return (
    <div style={{ background: '#0A0A0A', minHeight: '100vh', color: '#E0E0E0', fontFamily: 'Helvetica, Arial, sans-serif' }}>
      {/* Global Styles */}
      <style>{`
        body {
            background-color: #0A0A0A;
            margin: 0;
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .fade-in {
          animation: fadeIn 0.8s ease-out forwards;
        }
        @keyframes waveGradient {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        .wave-gradient {
          background: linear-gradient(90deg, #FFFFFF, #D4AF37, #FFFFFF, #D4AF37, #FFFFFF);
          background-size: 250% auto;
          animation: waveGradient 8s ease-in-out infinite;
        }
      `}</style>
      
      {/* Navigation Bar */}
      <nav style={{ background: '#0A0A0A', padding: '20px 60px', display: 'flex', justifyContent: 'center', gap: '100px', alignItems: 'center' }}>
        <div style={{ display: 'flex', gap: '40px', alignItems: 'center' }}>
          <button onClick={() => setPage('home')} style={{ background: 'none', border: 'none', color: '#A0A0A0', cursor: 'pointer', fontSize: '14px', transition: 'color 0.2s' }} onMouseEnter={(e) => e.target.style.color = '#fff'} onMouseLeave={(e) => e.target.style.color = '#A0A0A0'}>Home</button>
          <button style={{ background: 'none', border: 'none', color: '#A0A0A0', cursor: 'pointer', fontSize: '14px', transition: 'color 0.2s' }} onMouseEnter={(e) => e.target.style.color = '#fff'} onMouseLeave={(e) => e.target.style.color = '#A0A0A0'}>About</button>
        </div>
        <div className="wave-gradient" style={{ fontSize: '22px', fontWeight: '500', letterSpacing: '1px', 
                    WebkitBackgroundClip: 'text', 
                    WebkitTextFillColor: 'transparent',
                    backgroundClip: 'text',
                    color: 'transparent' // Fallback for browsers that don't support text-fill-color
                }}>AutoPharmaX</div>
        <div style={{ display: 'flex', gap: '40px', alignItems: 'center' }}>
          <button style={{ background: 'none', border: 'none', color: '#A0A0A0', cursor: 'pointer', fontSize: '14px', transition: 'color 0.2s' }} onMouseEnter={(e) => e.target.style.color = '#fff'} onMouseLeave={(e) => e.target.style.color = '#A0A0A0'}>Github</button>
          <button style={{ background: 'none', border: 'none', color: '#A0A0A0', cursor: 'pointer', fontSize: '14px', transition: 'color 0.2s' }} onMouseEnter={(e) => e.target.style.color = '#fff'} onMouseLeave={(e) => e.target.style.color = '#A0A0A0'}>Contact</button>
        </div>
      </nav>

      {/* Main Content */}
      {page === 'home' && (
        <main className="fade-in">
          {/* Hero Section */}
          <section style={{ minHeight: 'calc(100vh - 80px)', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', textAlign: 'center', padding: '80px 40px', position: 'relative', overflow: 'hidden' }}>
            <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, background: 'radial-gradient(ellipse at center, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0) 70%)', pointerEvents: 'none' }}></div>
            {/* Background Lines SVG */}
            <svg width="100%" height="100%" style={{ position: 'absolute', top: 0, left: 0, opacity: 0.1, pointerEvents: 'none' }} >
              <defs>
                <linearGradient id="lineGrad" x1="0%" y1="50%" x2="100%" y2="50%">
                  <stop offset="0%" stopColor="#D4AF37" stopOpacity="0" />
                  <stop offset="50%" stopColor="#D4AF37" stopOpacity="1" />
                  <stop offset="100%" stopColor="#D4AF37" stopOpacity="0" />
                </linearGradient>
              </defs>
              {[...Array(10)].map((_, i) => (
                <path
                  key={i}
                  d={`M -100,${100 + i * 80} C 400,${-50 + i * 80} 800,${250 + i * 80} 1400,${100 + i * 80}`}
                  stroke="url(#lineGrad)"
                  strokeWidth="0.5"
                  fill="none"
                />
              ))}
            </svg>

            <div style={{ position: 'relative', zIndex: 1, maxWidth: '900px' }}>
              <h1 style={{ fontSize: '56px', fontWeight: '400', letterSpacing: '-1.5px', marginBottom: '24px', lineHeight: 1.2, 
                background: 'linear-gradient(90deg, #FFFFFF, #999999)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
                color: 'transparent'
              }}>AI-Powered Drug Efficacy Prediction<br />for Personalized Oncology</h1>
              <p style={{ fontSize: '18px', color: '#A0A0A0', marginBottom: '48px', lineHeight: 1.7, maxWidth: '700px', margin: '0 auto 48px' }}>Instantly predict drug response with high accuracy. Accelerate your research, reduce costs, and unlock the future of personalized medicine.</p>
              <button onClick={() => setPage('predict')} style={{ padding: '12px 24px', border: '1px solid #A0A0A0', color: '#E0E0E0', borderRadius: '8px', fontSize: '16px', fontWeight: 500, background: 'transparent', cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: '8px', transition: 'all 0.3s' }} onMouseEnter={(e) => { e.currentTarget.style.borderColor = '#fff'; e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)'; }} onMouseLeave={(e) => { e.currentTarget.style.borderColor = '#A0A0A0'; e.currentTarget.style.background = 'transparent'; }}>
                Predict Now <ChevronRight size={20} />
              </button>
            </div>
          </section>

          {/* Why Us Section */}
          <section style={{ padding: '120px 80px', background: '#0A0A0A' }}>
            <h2 style={{ textAlign: 'center', fontSize: '48px', marginBottom: '80px', fontWeight: '600', letterSpacing: '-1px', color: '#FFFFFF' }}>Why Us?</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '40px', marginBottom: '40px' }}>
              {[
                { num: '1', title: 'Unmatched Accuracy', desc: 'Our fine-tuned XGBoost model achieves a 99.20% R² Score, ensuring predictions that closely mirror real-world lab outcomes.' },
                { num: '2', title: 'High Correlation', desc: 'We demonstrate a 99.62% Pearson correlation with actual experimental values, validating our model\'s predictive power.' },
                { num: '3', title: 'Impressive Low Error', desc: 'With a Root Mean Square Error (RMSE) of only 0.2512, our predictions are precise, reliable, and ready for critical research applications.' }
              ].map((feature, idx) => (
                <div key={idx} style={{ position: 'relative', textAlign: 'left', padding: '32px', border: '1px solid rgba(212, 175, 55, 0.2)', borderRadius: '12px', background: 'linear-gradient(145deg, rgba(212, 175, 55, 0.02), rgba(212, 175, 55, 0.05))', overflow: 'hidden' }}>
                  {/* Moved number to bottom right */}
                  <div style={{ position: 'absolute', bottom: '-40px', right: '10px', fontSize: '180px', fontWeight: '700', color: 'rgba(212, 175, 55, 0.08)', zIndex: 0, lineHeight: 1, pointerEvents: 'none' }}>{feature.num}</div>
                  <div style={{ position: 'relative', zIndex: 1 }}>
                    <h3 style={{ fontSize: '20px', fontWeight: '600', marginBottom: '12px', color: '#D4AF37' }}>{feature.title}</h3>
                    <p style={{ fontSize: '15px', color: '#A0A0A0', lineHeight: 1.6 }}>{feature.desc}</p>
                  </div>
                </div>
              ))}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '40px' }}>
               {[
                { num: '4', title: 'Production Ready & Scalable', desc: 'AutoPharmaX is a fully deployed application, built on a robust pipeline ready to integrate into real-world research workflows and scale with your demands.' },
                { num: '5', title: 'End-to-End Integrity', desc: 'Our predictions are powered by a comprehensive data pipeline, ensuring quality, consistency, and traceability.' },
                { num: '6', title: 'Best-in-Class Technology', desc: 'We rigorously tested multiple models. Our fine-tuned XGBoost was chosen for its demonstrably superior performance.' }
              ].map((feature, idx) => (
                 <div key={idx} style={{ position: 'relative', textAlign: 'left', padding: '32px', border: '1px solid rgba(212, 175, 55, 0.2)', borderRadius: '12px', background: 'linear-gradient(145deg, rgba(212, 175, 55, 0.02), rgba(212, 175, 55, 0.05))', overflow: 'hidden' }}>
                    <div style={{ position: 'absolute', bottom: '-40px', right: '10px', fontSize: '180px', fontWeight: '700', color: 'rgba(212, 175, 55, 0.08)', zIndex: 0, lineHeight: 1, pointerEvents: 'none' }}>{feature.num}</div>
                     <div style={{ position: 'relative', zIndex: 1 }}>
                         <h3 style={{ fontSize: '20px', fontWeight: '600', marginBottom: '12px', color: '#D4AF37' }}>{feature.title}</h3>
                         <p style={{ fontSize: '15px', color: '#A0A0A0', lineHeight: 1.6 }}>{feature.desc}</p>
                     </div>
                 </div>
              ))}
            </div>
          </section>
        </main>
      )}

      {/* Prediction Page */}
      {page === 'predict' && (
        <section className="fade-in" style={{ minHeight: 'calc(100vh - 80px)', padding: '60px 80px', background: '#0A0A0A' }}>
          <button onClick={() => setPage('home')} style={{ marginBottom: '40px', display: 'flex', alignItems: 'center', gap: '8px', background: 'none', border: '1px solid #A0A0A0', color: '#E0E0E0', padding: '10px 20px', borderRadius: '8px', cursor: 'pointer', fontSize: '14px', transition: 'all 0.3s' }} onMouseEnter={(e) => { e.currentTarget.style.borderColor = '#fff'; e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)'; }} onMouseLeave={(e) => { e.currentTarget.style.borderColor = '#A0A0A0'; e.currentTarget.style.background = 'transparent'; }}>
            <ArrowLeft size={16} /> Back to Home
          </button>

          <div style={{ maxWidth: '800px', margin: '0 auto', padding: '60px', background: 'rgba(255, 255, 255, 0.02)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '12px' }}>
            <h1 style={{ fontSize: '42px', marginBottom: '40px', textAlign: 'center', fontWeight: 600, color: '#FFFFFF' }}>Drug Response Prediction</h1>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', marginBottom: '40px' }}>
              <div>
                <label style={{ display: 'block', marginBottom: '10px', fontSize: '14px', fontWeight: '500', color: '#D4AF37' }}>Select Cancer Cell Line</label>
                <select value={selectedCellLine} onChange={(e) => setSelectedCellLine(e.target.value)} style={{ width: '100%', padding: '12px', background: 'rgba(0, 0, 0, 0.3)', border: '1px solid #444', borderRadius: '8px', color: '#fff', fontSize: '16px' }}>
                  {cellLines.map(line => <option key={line} value={line}>{line}</option>)}
                </select>
              </div>
              <div>
                <label style={{ display: 'block', marginBottom: '10px', fontSize: '14px', fontWeight: '500', color: '#D4AF37' }}>Select Drug</label>
                <select value={selectedDrug} onChange={(e) => setSelectedDrug(e.target.value)} style={{ width: '100%', padding: '12px', background: 'rgba(0, 0, 0, 0.3)', border: '1px solid #444', borderRadius: '8px', color: '#fff', fontSize: '16px' }}>
                  {drugs.map(drug => <option key={drug} value={drug}>{drug}</option>)}
                </select>
              </div>
            </div>

            {error && <div style={{ color: '#ff8a8a', marginBottom: '20px', padding: '12px', background: 'rgba(255, 107, 107, 0.1)', borderRadius: '8px', textAlign: 'center' }}>{error}</div>}

            <button onClick={handlePredict} disabled={loading} style={{ width: '100%', padding: '16px', background: '#D4AF37', color: '#0A0A0A', border: 'none', borderRadius: '8px', fontSize: '16px', fontWeight: '600', cursor: loading ? 'not-allowed' : 'pointer', transition: 'all 0.3s', opacity: loading ? 0.6 : 1 }}>
              {loading ? 'Predicting...' : 'Predict IC50'}
            </button>

            {result && (
              <div style={{ marginTop: '50px' }}>
                <h2 style={{ fontSize: '28px', marginBottom: '30px', fontWeight: 500, textAlign: 'center', color: '#FFFFFF' }}>Prediction Results</h2>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
                    <ResultCard value={result.predicted_ln_ic50?.toFixed(4)} label="Predicted LN(IC50)" />
                    <ResultCard value={result.predicted_ic50?.toFixed(4)} label="Predicted IC50 (µM)" />
                </div>
                {result.actual_ln_ic50 && (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '20px' }}>
                     <ResultCard value={result.actual_ln_ic50?.toFixed(4)} label="Actual LN(IC50)" />
                     <ResultCard value={result.actual_ic50?.toFixed(4)} label="Actual IC50 (µM)" />
                     <ResultCard value={result.absolute_error?.toFixed(4)} label="Absolute Error" />
                  </div>
                )}
              </div>
            )}
          </div>
        </section>
      )}
    </div>
  );
}

// Helper component for result cards on prediction page
const ResultCard = ({ value, label }) => (
    <div style={{ padding: '24px', background: 'rgba(212, 175, 55, 0.05)', border: '1px solid rgba(212, 175, 55, 0.1)', borderRadius: '8px', textAlign: 'center' }}>
        <div style={{ fontSize: '32px', fontWeight: '500', color: '#D4AF37', marginBottom: '8px' }}>{value}</div>
        <div style={{ fontSize: '12px', color: '#A0A0A0', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{label}</div>
    </div>
);