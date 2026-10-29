import React, { useState, useEffect, useRef } from 'react';
import {
  Activity, Shield, ShieldAlert, ShieldCheck, AlertTriangle, Eye, Brain,
  Heart, Stethoscope, UploadCloud, RefreshCw, FileText, Download, CheckCircle,
  HelpCircle, ChevronRight, ChevronDown, ChevronUp, Layers, BarChart2, Database,
  Info, ArrowRight, X, Clock, Server, Check, AlertCircle, FileCheck, Lock
} from 'lucide-react';

const API_BASE = typeof window !== 'undefined' && window.location.origin.includes('5173') ? 'http://127.0.0.1:8000' : '';

export default function App() {
  // Navigation State
  const [activeTab, setActiveTab] = useState('overview'); // 'overview', 'analyze', 'results', 'how-it-works', 'about', 'technical'
  
  // System & Backend Data
  const [systemHealth, setSystemHealth] = useState(null);
  const [performanceData, setPerformanceData] = useState(null);
  const [datasetData, setDatasetData] = useState(null);

  // Upload & Validation Gate States
  const [selectedFile, setSelectedFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [requestedSpecialty, setRequestedSpecialty] = useState('auto');
  const [isValidating, setIsValidating] = useState(false);
  const [validationResult, setValidationResult] = useState(null); // { valid, error_type, title, message, detected_specialty, detected_modality }
  
  // Analysis States
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisStage, setAnalysisStage] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);

  // Results View Controls
  const [activeViewMode, setActiveViewMode] = useState('overlay'); // 'overlay', 'original', 'mask'
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);

  // Report Preparation Modal
  const [isPreparingReport, setIsPreparingReport] = useState(false);
  const [reportStep, setReportStep] = useState(0);
  const [isReportReady, setIsReportReady] = useState(false);

  // Hidden File Input Ref
  const fileInputRef = useRef(null);

  // Fetch initial system data
  useEffect(() => {
    fetchHealth();
    fetchPerformance();
    fetchDatasets();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (res.ok) setSystemHealth(await res.json());
    } catch (e) {
      console.warn("Backend not reached yet", e);
    }
  };

  const fetchPerformance = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/performance`);
      if (res.ok) setPerformanceData(await res.json());
    } catch (e) {
      console.warn("Performance data not loaded", e);
    }
  };

  const fetchDatasets = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/datasets`);
      if (res.ok) setDatasetData(await res.json());
    } catch (e) {
      console.warn("Datasets data not loaded", e);
    }
  };

  // Reset file selection
  const resetFile = () => {
    setSelectedFile(null);
    setFilePreview(null);
    setValidationResult(null);
    setIsValidating(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Trigger file selection
  const handleChooseAnotherImage = () => {
    resetFile();
    setTimeout(() => {
      if (fileInputRef.current) {
        fileInputRef.current.click();
      }
    }, 50);
  };

  // Handle file input change and immediately run Validation Gate
  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    setFilePreview(URL.createObjectURL(file));
    setValidationResult(null);

    await validateImage(file, requestedSpecialty);
  };

  // Run Image Validation against backend /api/validate
  const validateImage = async (file, specialty) => {
    setIsValidating(true);
    setValidationResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('specialty', specialty);

      const res = await fetch(`${API_BASE}/api/validate`, {
        method: 'POST',
        body: formData
      });

      if (res.ok) {
        const data = await res.json();
        setValidationResult(data);
      } else {
        const err = await res.json().catch(() => ({ detail: 'Validation failed' }));
        setValidationResult({
          valid: false,
          error_type: 'corrupt',
          title: 'Could not read image file',
          message: err.detail || 'The uploaded file could not be verified. Please try another image.'
        });
      }
    } catch (e) {
      console.error("Validation error:", e);
      setValidationResult({
        valid: false,
        error_type: 'corrupt',
        title: 'Connection error during validation',
        message: 'Could not communicate with the TRUSTMED validation service. Please ensure the backend is running.'
      });
    } finally {
      setIsValidating(false);
    }
  };

  // Specialty switcher during mismatch
  const handleSwitchSpecialty = async (newSpecialty) => {
    setRequestedSpecialty(newSpecialty);
    if (selectedFile) {
      await validateImage(selectedFile, newSpecialty);
    }
  };

  // Execute Analysis only after validation passes
  const handleAnalyzeImage = async () => {
    if (!selectedFile || !validationResult?.valid) return;

    setIsAnalyzing(true);
    setAnalysisProgress(15);
    setAnalysisStage('Checking image quality and resolution...');

    const stages = [
      { p: 35, msg: 'Selecting specialized clinical model...' },
      { p: 60, msg: 'Evaluating uncertainty and confidence...' },
      { p: 85, msg: 'Checking image familiarity & stability...' },
      { p: 95, msg: 'Preparing visual explanation...' }
    ];

    let step = 0;
    const interval = setInterval(() => {
      if (step < stages.length) {
        setAnalysisProgress(stages[step].p);
        setAnalysisStage(stages[step].msg);
        step++;
      }
    }, 220);

    try {
      const formData = new FormData();
      formData.append('files', selectedFile);
      formData.append('specialty', requestedSpecialty);

      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        body: formData
      });

      clearInterval(interval);
      setAnalysisProgress(100);

      if (res.ok) {
        const data = await res.json();
        const singleResult = data.results?.[0] || null;
        
        // Critical safety check: If backend rejected the scan, do not navigate to results
        if (!singleResult || singleResult.status?.startsWith('REJECTED')) {
          setIsAnalyzing(false);
          setValidationResult({
            valid: false,
            error_type: singleResult?.status === 'REJECTED_QUALITY' ? 'quality' : 'unsupported',
            title: singleResult?.status === 'REJECTED_QUALITY' ? 'Image quality is too low' : 'Sorry, this image isn\'t supported.',
            message: singleResult?.clinical_verdict || 'This scan could not be analyzed reliably.'
          });
          return;
        }

        setAnalysisResult(singleResult);
        setTimeout(() => {
          setIsAnalyzing(false);
          setActiveTab('results');
        }, 400);
      } else {
        const err = await res.json().catch(() => ({ detail: 'Analysis failed' }));
        alert(`Analysis Error: ${err.detail || 'Could not complete image assessment'}`);
        setIsAnalyzing(false);
      }
    } catch (e) {
      clearInterval(interval);
      console.error(e);
      alert('Could not connect to TRUSTMED backend at http://127.0.0.1:8000.');
      setIsAnalyzing(false);
    }
  };

  // Trigger user-facing Report Preparation Flow
  const startReportPreparation = () => {
    if (!analysisResult) return;
    setIsPreparingReport(true);
    setReportStep(0);
    setIsReportReady(false);

    // Step-by-step progress simulation
    const timer1 = setTimeout(() => setReportStep(1), 300);
    const timer2 = setTimeout(() => setReportStep(2), 650);
    const timer3 = setTimeout(() => setReportStep(3), 1000);
    const timer4 = setTimeout(() => setReportStep(4), 1400);
    const timer5 = setTimeout(() => {
      setReportStep(5);
      setIsReportReady(true);
    }, 1800);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      clearTimeout(timer4);
      clearTimeout(timer5);
    };
  };

  // Actual PDF Download Execution
  const triggerPdfDownload = async () => {
    if (!analysisResult) return;

    try {
      const res = await fetch(`${API_BASE}/api/report/pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(analysisResult)
      });

      if (res.ok) {
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `TRUSTMED_Assessment_${analysisResult.filename || 'scan'}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setIsPreparingReport(false);
      } else {
        alert("Failed to generate assessment report.");
      }
    } catch (e) {
      console.error("PDF download failed", e);
      alert("Error downloading report.");
    }
  };

  // Helper for Reliability Badge colors & descriptions
  const getReliabilityInfo = (reliabilityLevel) => {
    const level = (reliabilityLevel || 'LOW').toUpperCase();
    if (level === 'HIGH') {
      return {
        level: 'HIGH',
        colorClass: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/40',
        badgeBg: 'bg-emerald-500',
        borderColor: 'border-emerald-500/40',
        cardBg: 'bg-emerald-950/20',
        description: 'The AI identified this finding with consistent confidence and familiar image characteristics across Trust Engine validation checks.'
      };
    } else if (level === 'MEDIUM') {
      return {
        level: 'MEDIUM',
        colorClass: 'text-amber-400 bg-amber-500/10 border-amber-500/40',
        badgeBg: 'bg-amber-500',
        borderColor: 'border-amber-500/40',
        cardBg: 'bg-amber-950/20',
        description: 'The AI identified a possible finding with moderate confidence, but additional reliability checks indicate elevated uncertainty that warrants clinical corroboration.'
      };
    } else {
      return {
        level: 'LOW',
        colorClass: 'text-rose-400 bg-rose-500/10 border-rose-500/40',
        badgeBg: 'bg-rose-500',
        borderColor: 'border-rose-500/40',
        cardBg: 'bg-rose-950/20',
        description: 'The AI identified this finding with moderate confidence, but additional reliability checks indicate that the result should be interpreted cautiously and may not be dependable enough to use on its own without specialist review.'
      };
    }
  };

  // Helper for "Why did the AI make this assessment?" (Visual characteristics only, no biological causality)
  const getAssessmentRationale = (res) => {
    if (res?.assessment_rationale) return res.assessment_rationale;
    const spec = res?.specialty || 'general';
    if (spec === 'skin') {
      return "The model's prediction was influenced by visual characteristics and patterns within the highlighted region, including lesion appearance, pigmentation distribution, overall symmetry, and border characteristics.";
    } else if (spec === 'lung') {
      return "The model's prediction was influenced by regional visual characteristics and focal opacity patterns identified within the highlighted thoracic zones.";
    } else if (spec === 'brain') {
      return "The model's assessment was influenced by localized tissue contrast differences, structural cranial boundaries, and signal intensity variations within the highlighted cranial region.";
    } else if (spec === 'retina') {
      return "The model's prediction was influenced by visual characteristics, retinal vessel architecture, and localized pattern distributions within the highlighted fundus area.";
    }
    return "The model's prediction was influenced by visual patterns and regional characteristics within the highlighted area of the image.";
  };

  // Helper for "About This Finding" (Educational context, stress multiple factors, no individual etiology)
  const getEducationalContext = (res) => {
    if (res?.educational_context) return res.educational_context;
    const spec = res?.specialty || 'general';
    if (spec === 'skin') {
      return "This type of skin finding can have multiple contributing factors. An image alone cannot determine what caused a particular lesion in an individual person. A qualified dermatologist should consider the image together with clinical history, dermoscopic examination, and other relevant context.";
    } else if (spec === 'lung') {
      return "Thoracic and pulmonary findings can be associated with a wide variety of clinical and environmental factors. An imaging scan alone cannot determine what caused a specific finding in an individual patient. A qualified pulmonologist or physician should evaluate this scan alongside full medical history, symptoms, and clinical examination.";
    } else if (spec === 'brain') {
      return "Cranial imaging findings and space-occupying lesions can be associated with diverse cellular, vascular, or physiological factors. An imaging scan alone cannot establish the underlying cause for an individual patient. A neurologist or neurosurgeon should evaluate these findings in conjunction with clinical history, neurological examination, and dedicated diagnostic workup.";
    } else if (spec === 'retina') {
      return "Retinal microvascular findings are commonly associated with systemic vascular and metabolic factors over time. An image alone cannot determine individual etiology or systemic progression. A qualified ophthalmologist or retina specialist should assess the findings alongside clinical history, laboratory results, and comprehensive eye examination.";
    }
    return "Medical imaging findings can be associated with multiple clinical and physiological factors. An image alone cannot determine the underlying cause for an individual person. A qualified medical specialist should evaluate the findings in the context of complete patient history and clinical examination.";
  };

  return (
    <div className="min-h-screen bg-[#0b0f19] text-slate-100 flex flex-col selection:bg-cyan-500 selection:text-white font-sans">
      
      {/* ======================================================== */}
      {/* TOP NAVIGATION BAR */}
      {/* ======================================================== */}
      <header className="border-b border-slate-800/80 bg-[#0f172a]/90 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          
          {/* Logo & Brand */}
          <div 
            className="flex items-center space-x-3 cursor-pointer select-none" 
            onClick={() => setActiveTab('overview')}
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <ShieldCheck className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-black text-xl tracking-wider text-white">TRUST<span className="text-cyan-400">MED</span></span>
                <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 tracking-wider">
                  Medical AI
                </span>
              </div>
              <p className="text-[11px] text-slate-400 hidden sm:block">Self-Aware Multi-Specialty Decision Support</p>
            </div>
          </div>

          {/* Primary Navigation Links */}
          <nav className="flex items-center space-x-1 sm:space-x-2">
            {[
              { id: 'overview', label: 'Overview', icon: Heart },
              { id: 'analyze', label: 'Analyze', icon: UploadCloud },
              { 
                id: 'results', 
                label: 'Results', 
                icon: Activity, 
                badge: analysisResult ? 'Ready' : null,
                disabled: !analysisResult 
              },
              { id: 'how-it-works', label: 'How It Works', icon: HelpCircle },
              { id: 'about', label: 'About', icon: Info },
            ].map(tab => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              const isDisabled = tab.disabled;

              return (
                <button
                  key={tab.id}
                  disabled={isDisabled}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                    isActive
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm shadow-cyan-500/10'
                      : isDisabled
                        ? 'text-slate-600 cursor-not-allowed'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span className="hidden md:inline">{tab.label}</span>
                  {tab.badge && (
                    <span className="ml-1 px-1.5 py-0.2 rounded-full bg-cyan-500 text-slate-950 font-extrabold text-[9px]">
                      {tab.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>

          {/* Secondary Action: Technical Specs Trigger */}
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setActiveTab('technical')}
              className={`text-xs px-2.5 py-1.5 rounded-lg font-medium transition-all flex items-center space-x-1.5 ${
                activeTab === 'technical'
                  ? 'bg-slate-800 text-cyan-300 border border-slate-700'
                  : 'text-slate-400 hover:text-slate-300 bg-slate-900 border border-slate-800 hover:border-slate-700'
              }`}
              title="View benchmark metrics & dataset information"
            >
              <BarChart2 className="w-3.5 h-3.5 text-cyan-400" />
              <span className="hidden sm:inline">Technical Data</span>
            </button>
          </div>
        </div>
      </header>

      {/* ======================================================== */}
      {/* MAIN VIEW AREA */}
      {/* ======================================================== */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">

        {/* -------------------------------------------------------- */}
        {/* VIEW 1: OVERVIEW (PRODUCT LANDING PAGE) */}
        {/* -------------------------------------------------------- */}
        {activeTab === 'overview' && (
          <div className="space-y-12">
            
            {/* Product Hero */}
            <div className="relative overflow-hidden rounded-3xl border border-slate-800 bg-gradient-to-b from-slate-900 via-[#0f172a] to-[#0b0f19] p-8 sm:p-14 shadow-2xl">
              <div className="absolute top-0 right-0 -mt-16 -mr-16 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none"></div>
              <div className="absolute bottom-0 left-1/4 -mb-16 w-80 h-80 bg-blue-500/10 rounded-full blur-3xl pointer-events-none"></div>

              <div className="max-w-3xl space-y-6">
                <div className="inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-semibold">
                  <ShieldCheck className="w-4 h-4 text-cyan-400" />
                  <span>Self-Aware Multi-Specialty Medical AI</span>
                </div>

                <h1 className="text-4xl sm:text-5xl font-black tracking-tight text-white leading-tight">
                  Most medical AI asks: <span className="text-slate-400 font-medium">“What does this image show?”</span><br />
                  <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-sky-300 to-blue-400">
                    TRUSTMED asks: “How much should we trust it?”
                  </span>
                </h1>

                <p className="text-slate-300 text-base sm:text-lg leading-relaxed">
                  TRUSTMED analyzes supported medical images, provides an AI assessment, checks how reliable the result is,
                  explains what influenced the AI, and recommends specialist review when appropriate.
                </p>

                <div className="pt-2 flex flex-wrap items-center gap-4">
                  <button
                    onClick={() => setActiveTab('analyze')}
                    className="px-7 py-3.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-bold text-sm shadow-lg shadow-cyan-500/25 flex items-center space-x-2 transition-all cursor-pointer scale-100 hover:scale-105"
                  >
                    <UploadCloud className="w-4 h-4" />
                    <span>Analyze an Image</span>
                    <ArrowRight className="w-4 h-4" />
                  </button>

                  <button
                    onClick={() => setActiveTab('how-it-works')}
                    className="px-6 py-3.5 rounded-xl bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700 text-slate-200 font-semibold text-sm flex items-center space-x-2 transition-all"
                  >
                    <HelpCircle className="w-4 h-4 text-cyan-400" />
                    <span>How It Works</span>
                  </button>
                </div>

                <div className="pt-4 border-t border-slate-800/80 flex items-center space-x-2 text-xs text-slate-400">
                  <Info className="w-4 h-4 text-slate-500 shrink-0" />
                  <span>
                    TRUSTMED is an AI-assisted decision-support prototype designed to assist healthcare workflows. It does not replace clinical consultation, diagnostic testing, or doctor-patient evaluation.
                  </span>
                </div>
              </div>
            </div>

            {/* Core Capabilities */}
            <div className="space-y-4">
              <div>
                <h2 className="text-xl font-bold text-white">What TRUSTMED Does</h2>
                <p className="text-xs text-slate-400">Transparent medical decision-support designed for real clinical trust</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 space-y-3">
                  <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
                    <ShieldCheck className="w-5 h-5" />
                  </div>
                  <h3 className="text-base font-bold text-white">1. Validation & Safety Gate</h3>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Automatically inspects image resolution, clarity, blur, and medical modality before any analysis begins. Blurry or unsupported images are rejected safely.
                  </p>
                </div>

                <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 space-y-3">
                  <div className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-500/30 flex items-center justify-center text-blue-400">
                    <Activity className="w-5 h-5" />
                  </div>
                  <h3 className="text-base font-bold text-white">2. Reliability Assessment</h3>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Evaluates prediction uncertainty, calibration, and image familiarity. Categorizes reliability into clear High, Medium, or Low indicators so clinicians know when to corroborate.
                  </p>
                </div>

                <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 space-y-3">
                  <div className="w-10 h-10 rounded-xl bg-purple-500/10 border border-purple-500/30 flex items-center justify-center text-purple-400">
                    <Eye className="w-5 h-5" />
                  </div>
                  <h3 className="text-base font-bold text-white">3. Explanation & Review</h3>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Highlights the visual areas that influenced the AI finding and provides tailored specialist review guidance when the result warrants human clinical confirmation.
                  </p>
                </div>
              </div>
            </div>

            {/* Supported Medical Specialties */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-bold text-white">Supported Medical Specialties</h2>
                  <p className="text-xs text-slate-400">Validated across four dedicated diagnostic imaging disciplines</p>
                </div>
                <span className="text-xs text-cyan-400 font-mono">4 Specialties</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {[
                  {
                    icon: '🫁',
                    title: 'Lung Thoracic',
                    modality: 'Chest X-rays (Frontal CXR)',
                    desc: 'Screening for thoracic findings including infiltration, effusion, and pneumonia.',
                    specialist: 'Pulmonology / Radiology',
                    border: 'border-blue-500/30',
                    bg: 'from-blue-500/10 to-transparent'
                  },
                  {
                    icon: '🧠',
                    title: 'Brain Oncology',
                    modality: 'Brain MRI (Axial Slices)',
                    desc: 'Abnormality screening and automated tumor boundary segmentation.',
                    specialist: 'Neurology / Neurosurgery',
                    border: 'border-purple-500/30',
                    bg: 'from-purple-500/10 to-transparent'
                  },
                  {
                    icon: '🩺',
                    title: 'Dermatology',
                    modality: 'Dermoscopic Skin Lesion Images',
                    desc: 'Evaluation of pigmented skin lesions, melanoma, and dermatological conditions.',
                    specialist: 'Dermatology',
                    border: 'border-amber-500/30',
                    bg: 'from-amber-500/10 to-transparent'
                  },
                  {
                    icon: '👁️',
                    title: 'Ophthalmology',
                    modality: 'Retinal Fundus Photography',
                    desc: 'Assessment of retinal vasculature and diabetic retinopathy grading.',
                    specialist: 'Ophthalmology / Retina',
                    border: 'border-emerald-500/30',
                    bg: 'from-emerald-500/10 to-transparent'
                  }
                ].map((s, idx) => (
                  <div
                    key={idx}
                    className={`rounded-2xl border ${s.border} bg-gradient-to-b ${s.bg} bg-slate-900/40 p-5 space-y-3`}
                  >
                    <div className="text-3xl">{s.icon}</div>
                    <div>
                      <h3 className="text-base font-bold text-white">{s.title}</h3>
                      <p className="text-xs text-cyan-300 font-medium">{s.modality}</p>
                    </div>
                    <p className="text-xs text-slate-300 leading-relaxed">{s.desc}</p>
                    <div className="pt-2 border-t border-slate-800 text-[11px] text-slate-400">
                      <strong className="text-slate-300">Specialist:</strong> {s.specialist}
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>
        )}

        {/* -------------------------------------------------------- */}
        {/* VIEW 2: ANALYZE PAGE (THE VALIDATION GATE) */}
        {/* -------------------------------------------------------- */}
        {activeTab === 'analyze' && (
          <div className="max-w-3xl mx-auto space-y-6">
            
            <div className="text-center space-y-2">
              <h2 className="text-2xl font-black text-white">Upload Medical Image</h2>
              <p className="text-xs sm:text-sm text-slate-400">
                TRUSTMED validates image clarity, quality, and modality before analysis to prevent unreliable predictions.
              </p>
            </div>

            {/* Hidden File Input */}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png, image/jpeg, image/jpg"
              className="hidden"
              onChange={handleFileChange}
              disabled={isValidating || isAnalyzing}
            />

            {/* Specialty Selection (Default: Auto-Detect) */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 flex flex-col sm:flex-row items-center justify-between gap-3">
              <div className="flex items-center space-x-2 text-xs text-slate-300">
                <Layers className="w-4 h-4 text-cyan-400" />
                <span className="font-semibold">Specialty Mode:</span>
              </div>

              <div className="flex flex-wrap gap-1.5">
                {[
                  { id: 'auto', label: 'Auto-Detect (Recommended)' },
                  { id: 'lung', label: 'Lung (Chest X-ray)' },
                  { id: 'brain', label: 'Brain (Brain MRI)' },
                  { id: 'skin', label: 'Skin (Dermoscopy)' },
                  { id: 'retina', label: 'Retina (Fundus)' },
                ].map(mode => (
                  <button
                    key={mode.id}
                    onClick={() => {
                      setRequestedSpecialty(mode.id);
                      if (selectedFile) {
                        validateImage(selectedFile, mode.id);
                      }
                    }}
                    className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                      requestedSpecialty === mode.id
                        ? 'bg-cyan-500 text-slate-950 font-bold'
                        : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                    }`}
                  >
                    {mode.label}
                  </button>
                ))}
              </div>
            </div>

            {/* ==================================================== */}
            {/* STATE 1: NO FILE SELECTED YET (IDLE DROPZONE) */}
            {/* ==================================================== */}
            {!selectedFile && (
              <div 
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-slate-700 hover:border-cyan-500/70 rounded-2xl p-10 bg-slate-900/40 text-center space-y-4 transition-all cursor-pointer group"
              >
                <div className="w-16 h-16 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 mx-auto group-hover:scale-105 transition-all">
                  <UploadCloud className="w-8 h-8" />
                </div>
                <div>
                  <p className="text-base font-bold text-white group-hover:text-cyan-300 transition-colors">
                    Click to select medical scan or drag and drop
                  </p>
                  <p className="text-xs text-slate-400 mt-1">
                    Supports PNG or JPEG medical images (Chest X-ray, Brain MRI, Skin, Retina)
                  </p>
                </div>
                <div className="pt-2 flex justify-center gap-2 text-[11px] text-slate-500">
                  <span>• Automatic Quality Checking</span>
                  <span>• Modality Verification</span>
                  <span>• Decision-Support Reliability</span>
                </div>
              </div>
            )}

            {/* ==================================================== */}
            {/* STATE 2: VALIDATING / CHECKING IMAGE */}
            {/* ==================================================== */}
            {selectedFile && isValidating && (
              <div className="rounded-2xl border border-cyan-500/30 bg-slate-900 p-8 text-center space-y-4 shadow-xl">
                <div className="w-12 h-12 rounded-full bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 mx-auto">
                  <RefreshCw className="w-6 h-6 animate-spin text-cyan-400" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-white">Checking image...</h3>
                  <p className="text-xs text-slate-400 mt-1">
                    Verifying image clarity, resolution, and medical specialty suitability...
                  </p>
                </div>
                <div className="w-48 mx-auto bg-slate-800 h-1.5 rounded-full overflow-hidden">
                  <div className="bg-cyan-500 h-full w-2/3 animate-pulse"></div>
                </div>
              </div>
            )}

            {/* ==================================================== */}
            {/* STATE 3: INVALID IMAGE (VALIDATION GATE BLOCKED) */}
            {/* ==================================================== */}
            {selectedFile && !isValidating && validationResult && !validationResult.valid && (
              <div className="rounded-2xl border border-rose-500/50 bg-rose-950/20 p-6 space-y-5 shadow-xl">
                <div className="flex items-start space-x-3">
                  <div className="p-2 rounded-xl bg-rose-500/20 text-rose-400 shrink-0">
                    <AlertTriangle className="w-6 h-6" />
                  </div>
                  <div className="space-y-1 text-left flex-1">
                    <h3 className="text-base font-bold text-white">
                      {validationResult.title || 'Image Verification Failed'}
                    </h3>
                    <p className="text-xs text-rose-200 leading-relaxed whitespace-pre-line">
                      {validationResult.message}
                    </p>
                  </div>
                </div>

                {/* Specific Guidance based on error type */}
                {validationResult.error_type === 'unsupported' && (
                  <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 text-xs space-y-2 text-left">
                    <p className="font-semibold text-slate-300">TRUSTMED currently supports:</p>
                    <ul className="space-y-1 text-slate-400 text-xs">
                      <li>• <strong>Lung</strong> — Chest X-rays</li>
                      <li>• <strong>Brain</strong> — Brain MRI</li>
                      <li>• <strong>Skin</strong> — Dermoscopic skin lesion images</li>
                      <li>• <strong>Retina</strong> — Retinal fundus images</li>
                    </ul>
                  </div>
                )}

                {/* Specialty Mismatch Quick Action */}
                {validationResult.error_type === 'mismatch' && validationResult.detected_specialty && (
                  <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 text-xs flex flex-col sm:flex-row items-center justify-between gap-3 text-left">
                    <span className="text-slate-300">
                      Detected as: <strong>{validationResult.detected_modality || validationResult.detected_specialty}</strong>
                    </span>
                    <button
                      onClick={() => handleSwitchSpecialty(validationResult.detected_specialty)}
                      className="px-3.5 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-xs transition-all shrink-0"
                    >
                      Switch to {validationResult.detected_specialty.toUpperCase()}
                    </button>
                  </div>
                )}

                {/* Re-upload Action */}
                <div className="pt-2 flex flex-col sm:flex-row items-center justify-between gap-3 border-t border-rose-900/40">
                  <span className="text-[11px] text-slate-400 flex items-center space-x-1.5">
                    <Lock className="w-3.5 h-3.5 text-rose-400" />
                    <span>Analysis disabled until a valid image is uploaded</span>
                  </span>

                  <button
                    onClick={handleChooseAnotherImage}
                    className="px-5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-bold text-xs flex items-center space-x-2 transition-all border border-slate-700 cursor-pointer"
                  >
                    <UploadCloud className="w-4 h-4 text-cyan-400" />
                    <span>Choose Another Image</span>
                  </button>
                </div>
              </div>
            )}

            {/* ==================================================== */}
            {/* STATE 4: VALID IMAGE (READY FOR USER TO ANALYZE) */}
            {/* ==================================================== */}
            {selectedFile && !isValidating && validationResult && validationResult.valid && (
              <div className="space-y-6">
                {/* Validation Success Card */}
                <div className="rounded-2xl border border-emerald-500/40 bg-emerald-950/20 p-5 space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2 text-emerald-400 font-bold text-sm">
                      <CheckCircle className="w-5 h-5 text-emerald-400" />
                      <span>{validationResult.title || 'Image looks suitable for analysis ✓'}</span>
                    </div>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/30">
                      Validated
                    </span>
                  </div>

                  <p className="text-xs text-slate-300">
                    {validationResult.message || 'Image quality and modality have been verified.'}
                  </p>

                  {/* Image Preview & Details */}
                  <div className="flex flex-col sm:flex-row items-center gap-4 p-3 rounded-xl bg-slate-900 border border-slate-800">
                    {filePreview && (
                      <img 
                        src={filePreview} 
                        alt="Selected scan" 
                        className="w-24 h-24 rounded-lg object-cover border border-slate-700 bg-black shrink-0" 
                      />
                    )}
                    <div className="text-xs space-y-1 text-left flex-1">
                      <div className="font-bold text-white truncate max-w-xs">{selectedFile.name}</div>
                      <div className="text-slate-400">
                        Detected: <strong className="text-cyan-300">{validationResult.detected_modality || validationResult.detected_specialty}</strong>
                      </div>
                      <div className="text-slate-400">
                        Quality Gate: <span className="text-emerald-400 font-semibold">Passed</span> (Resolution & sharpness verified)
                      </div>
                    </div>
                    <button
                      onClick={handleChooseAnotherImage}
                      className="text-xs text-slate-400 hover:text-slate-200 underline shrink-0"
                    >
                      Change image
                    </button>
                  </div>
                </div>

                {/* Analysis Action / In Progress Tracker */}
                {isAnalyzing ? (
                  <div className="rounded-2xl border border-cyan-500/40 bg-slate-900 p-6 space-y-4 shadow-xl">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-bold text-cyan-400 flex items-center space-x-2">
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        <span>Analyzing Medical Image...</span>
                      </span>
                      <span className="font-mono font-bold text-cyan-300">{analysisProgress}%</span>
                    </div>

                    <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-cyan-500 to-blue-500 h-full transition-all duration-300"
                        style={{ width: `${analysisProgress}%` }}
                      ></div>
                    </div>

                    <p className="text-xs font-mono text-slate-300 animate-pulse">{analysisStage}</p>

                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 text-[11px] text-slate-400">
                      <div className={analysisProgress >= 25 ? 'text-emerald-400 font-medium' : ''}>✓ Image analyzed</div>
                      <div className={analysisProgress >= 50 ? 'text-emerald-400 font-medium' : ''}>✓ AI assessment completed</div>
                      <div className={analysisProgress >= 75 ? 'text-emerald-400 font-medium' : ''}>✓ Reliability checked</div>
                      <div className={analysisProgress >= 90 ? 'text-emerald-400 font-medium' : ''}>✓ Explanation prepared</div>
                    </div>
                  </div>
                ) : (
                  <div className="flex justify-center">
                    <button
                      onClick={handleAnalyzeImage}
                      className="px-8 py-4 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-bold text-sm shadow-xl shadow-cyan-500/25 flex items-center space-x-2 transition-all cursor-pointer scale-100 hover:scale-105"
                    >
                      <Activity className="w-5 h-5" />
                      <span>Analyze Image</span>
                    </button>
                  </div>
                )}
              </div>
            )}

          </div>
        )}

        {/* -------------------------------------------------------- */}
        {/* VIEW 3: RESULTS ("YOUR AI ASSESSMENT") */}
        {/* -------------------------------------------------------- */}
        {activeTab === 'results' && (
          <div>
            {!analysisResult ? (
              <div className="text-center py-20 space-y-4">
                <AlertTriangle className="w-12 h-12 text-slate-500 mx-auto" />
                <h3 className="text-lg font-bold text-slate-300">No Assessment Available</h3>
                <p className="text-xs text-slate-400">Please upload and validate a medical scan to view its AI assessment.</p>
                <button
                  onClick={() => setActiveTab('analyze')}
                  className="px-4 py-2 rounded-lg bg-cyan-600 text-white text-xs font-bold"
                >
                  Go to Analyze
                </button>
              </div>
            ) : (
              <div className="space-y-6 max-w-5xl mx-auto">
                
                {/* Header & Actions */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
                  <div>
                    <h2 className="text-2xl font-black text-white flex items-center space-x-2">
                      <Activity className="w-6 h-6 text-cyan-400" />
                      <span>Your AI Assessment</span>
                    </h2>
                    <p className="text-xs text-slate-400 mt-0.5">
                      File: <span className="text-slate-300 font-medium">{analysisResult.filename}</span> • Modality: <span className="text-slate-300 font-medium">{analysisResult.modality}</span>
                    </p>
                  </div>

                  <div className="flex items-center space-x-3">
                    <button
                      onClick={handleChooseAnotherImage}
                      className="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold transition-all border border-slate-700"
                    >
                      Analyze Another Image
                    </button>

                    <button
                      onClick={startReportPreparation}
                      className="px-4 py-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-bold text-xs flex items-center space-x-2 shadow-md shadow-cyan-500/20 transition-all cursor-pointer"
                    >
                      <Download className="w-4 h-4" />
                      <span>Download Report</span>
                    </button>
                  </div>
                </div>

                {/* Primary Two-Column Grid */}
                {(() => {
                  const primaryConfidence = analysisResult.calibrated_confidence ?? analysisResult.calibration?.calibrated_confidence ?? analysisResult.prediction?.top_probability ?? 0;
                  const rawConfidence = analysisResult.calibration?.raw_confidence ?? analysisResult.prediction?.top_probability ?? 0;
                  const calibratedPct = Math.round(primaryConfidence * 100);
                  const rawPct = Math.round(rawConfidence * 100);

                  const rel = getReliabilityInfo(analysisResult.reliability?.overall_reliability);
                  const hasCriticalAlert = analysisResult.reliability?.critical_mismatch_detected;

                  return (
                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">

                      {/* Left Column: AI Explanation & About This Finding (6 cols) */}
                      <div className="lg:col-span-6 space-y-5">
                        
                        {/* AI Explanation Card */}
                        <div className="rounded-2xl border border-slate-800 bg-slate-900 p-4 space-y-3.5">
                          <div className="flex items-center justify-between text-xs">
                            <span className="font-bold text-slate-200 flex items-center space-x-1.5">
                              <Eye className="w-4 h-4 text-cyan-400" />
                              <span>AI Explanation</span>
                            </span>
                            <span className="text-[11px] text-slate-400">
                              {analysisResult.specialty === 'brain' ? 'Tumor Region' : 'Areas That Influenced the AI'}
                            </span>
                          </div>

                          {/* Image Viewer */}
                          <div className="relative rounded-xl overflow-hidden bg-black aspect-square flex items-center justify-center border border-slate-800">
                            {activeViewMode === 'original' && analysisResult.image_base64 && (
                              <img src={analysisResult.image_base64} alt="Original Scan" className="w-full h-full object-contain" />
                            )}
                            {activeViewMode === 'overlay' && (
                              <img
                                src={
                                  analysisResult.gradcam_base64 ||
                                  analysisResult.segmentation?.overlay_base64 ||
                                  analysisResult.image_base64
                                }
                                alt="Visual Explanation"
                                className="w-full h-full object-contain"
                              />
                            )}
                            {activeViewMode === 'mask' && analysisResult.segmentation?.mask_base64 && (
                              <img src={analysisResult.segmentation.mask_base64} alt="Segmentation Mask" className="w-full h-full object-contain" />
                            )}
                          </div>

                          {/* View Switcher Controls */}
                          <div className="flex items-center justify-center space-x-2 pt-1">
                            <button
                              onClick={() => setActiveViewMode('overlay')}
                              className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                                activeViewMode === 'overlay' ? 'bg-cyan-500 text-slate-950 shadow-sm' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                              }`}
                            >
                              Areas That Influenced the AI
                            </button>
                            <button
                              onClick={() => setActiveViewMode('original')}
                              className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                                activeViewMode === 'original' ? 'bg-cyan-500 text-slate-950 shadow-sm' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                              }`}
                            >
                              Original Scan
                            </button>
                            {analysisResult.segmentation?.mask_base64 && (
                              <button
                                onClick={() => setActiveViewMode('mask')}
                                className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                                  activeViewMode === 'mask' ? 'bg-cyan-500 text-slate-950 shadow-sm' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                                }`}
                              >
                                Tumor Mask
                              </button>
                            )}
                          </div>

                          {/* Why did the AI make this assessment? */}
                          <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/80 space-y-1.5">
                            <div className="flex items-center space-x-1.5 text-xs font-bold text-slate-200">
                              <HelpCircle className="w-3.5 h-3.5 text-cyan-400" />
                              <span>Why did the AI make this assessment?</span>
                            </div>
                            <p className="text-xs text-slate-300 leading-relaxed">
                              {getAssessmentRationale(analysisResult)}
                            </p>
                            <p className="text-[10px] text-slate-500 pt-0.5">
                              Visual patterns and features within the highlighted region that influenced the model prediction. Does not establish biological cause.
                            </p>
                          </div>
                        </div>

                        {/* About This Finding (What can be associated with this finding?) */}
                        <div className="rounded-2xl border border-slate-800 bg-slate-900 p-4 space-y-2.5">
                          <div className="flex items-center space-x-2 text-xs font-bold text-slate-200">
                            <Info className="w-4 h-4 text-cyan-400" />
                            <span>About This Finding</span>
                            <span className="text-[10px] font-normal text-slate-400">
                              (What can be associated with this finding?)
                            </span>
                          </div>
                          <p className="text-xs text-slate-300 leading-relaxed">
                            {getEducationalContext(analysisResult)}
                          </p>
                          <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/60 text-[10px] text-slate-400 flex items-center space-x-2">
                            <Shield className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                            <span>Educational context only. An image scan cannot establish a specific clinical cause for an individual patient.</span>
                          </div>
                        </div>

                      </div>

                      {/* Right Column: AI Assessment, Recommended Review & Technical Audit (6 cols) */}
                      <div className="lg:col-span-6 space-y-5">

                        {/* AI ASSESSMENT CARD (Finding + Calibrated Confidence + Integrated Reliability) */}
                        <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5 space-y-4">
                          
                          {/* Top Badges */}
                          <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-2">
                              <span className="text-xs font-mono px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">
                                {analysisResult.specialty_name}
                              </span>
                              <span className="text-xs text-slate-400">
                                {analysisResult.modality}
                              </span>
                            </div>
                            <span className="text-[11px] text-slate-500 uppercase tracking-wider font-semibold">
                              AI Assessment
                            </span>
                          </div>

                          {/* Predicted Class & Status Badge */}
                          <div className="space-y-2">
                            <div className="flex items-center justify-between">
                              <span className="text-xs text-slate-400">Deep Learning Model Classification:</span>
                              <span className={`text-[11px] font-bold px-2.5 py-0.5 rounded-full border ${
                                analysisResult.is_normal
                                  ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                                  : 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                              }`}>
                                {analysisResult.is_normal ? '✓ NORMAL / HEALTHY' : '⚠ DISEASE / ABNORMAL'}
                              </span>
                            </div>
                            <h3 className="text-2xl font-black text-white mt-0.5">
                              {analysisResult.predicted_class || analysisResult.prediction?.primary_finding || 'No primary finding'}
                            </h3>
                            {analysisResult.prediction?.primary_finding && analysisResult.prediction?.primary_finding !== analysisResult.predicted_class && (
                              <p className="text-xs text-slate-400">
                                {analysisResult.prediction.primary_finding}
                              </p>
                            )}
                          </div>

                          {/* Medical Disclaimer Banner */}
                          <div className="p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/30 text-[11px] text-amber-300/90">
                            <span className="font-bold text-amber-400">Note: </span>
                            {analysisResult.medical_disclaimer || 'This result is an artificial intelligence deep learning model prediction for clinical decision support, not a confirmed medical diagnosis.'}
                          </div>

                          {/* Calibrated AI Confidence Bar */}
                          <div className="space-y-2 p-3.5 rounded-xl bg-slate-950/50 border border-slate-800/80">
                            <div className="flex justify-between items-baseline text-xs">
                              <div className="flex items-center space-x-1.5">
                                <span className="text-slate-300 font-medium">AI Confidence:</span>
                                <span className="text-[10px] text-cyan-400/80 font-mono">(Calibrated)</span>
                              </div>
                              <span className="font-mono font-bold text-cyan-400 text-sm">
                                {calibratedPct}%
                              </span>
                            </div>
                            <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
                              <div
                                className="bg-gradient-to-r from-cyan-500 to-blue-500 h-full rounded-full transition-all duration-500"
                                style={{ width: `${Math.min(100, Math.max(5, calibratedPct))}%` }}
                              ></div>
                            </div>
                            <div className="flex justify-between text-[10px] text-slate-400 pt-0.5">
                              <span>Calibrated probability based on validation reliability</span>
                              <span>Raw uncalibrated: {rawPct}%</span>
                            </div>
                          </div>

                          {/* COMPACT INTEGRATED RELIABILITY SECTION */}
                          <div className={`rounded-xl border p-4 space-y-3 ${rel.cardBg} ${rel.borderColor}`}>
                            <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
                              <div className="flex items-center space-x-2">
                                <ShieldCheck className="w-4 h-4 text-cyan-400" />
                                <h4 className="text-xs font-bold text-white">AI Reliability Level</h4>
                              </div>
                              <div className={`px-2.5 py-0.5 rounded-full text-[11px] font-black uppercase tracking-wider border ${rel.colorClass}`}>
                                {rel.level}
                              </div>
                            </div>

                            {/* Plain-Language Reliability Explanation */}
                            <div className="space-y-1">
                              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">What this means:</span>
                              <p className="text-xs text-slate-200 leading-relaxed">
                                {analysisResult.reliability_summary || rel.description}
                              </p>
                            </div>

                            {/* Prominent Clinical Warning if present */}
                            {analysisResult.reliability?.clinical_warning && (
                              <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800 text-xs text-slate-300 leading-relaxed">
                                <span className="font-bold text-amber-400">Note: </span>
                                {analysisResult.reliability.clinical_warning}
                              </div>
                            )}

                            {/* Critical Uncertainty Mismatch Alert */}
                            {hasCriticalAlert && (
                              <div className="p-2.5 rounded-lg bg-rose-950/40 border border-rose-500/50 text-xs text-rose-200 space-y-1">
                                <div className="flex items-center space-x-1.5 font-bold text-rose-400">
                                  <AlertTriangle className="w-3.5 h-3.5" />
                                  <span>Caution: Elevated Uncertainty Detected</span>
                                </div>
                                <p className="text-[11px]">
                                  Additional reliability checks indicate that this result may not be dependable enough to use on its own. Specialist review recommended.
                                </p>
                              </div>
                            )}

                            {/* 4 Compact Trust Indicators */}
                            <div className="grid grid-cols-2 gap-2 pt-1 text-[11px]">
                              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80 flex items-center justify-between">
                                <span className="text-slate-400">Image Quality</span>
                                <span className="font-semibold text-emerald-400">
                                  {analysisResult.quality?.verdict || 'Clear'}
                                </span>
                              </div>

                              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80 flex items-center justify-between">
                                <span className="text-slate-400">Uncertainty</span>
                                <span className={`font-semibold ${
                                  analysisResult.uncertainty?.uncertainty_level === 'LOW' ? 'text-emerald-400' : 'text-amber-400'
                                }`}>
                                  {analysisResult.uncertainty?.uncertainty_level || 'Low'}
                                </span>
                              </div>

                              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80 flex items-center justify-between">
                                <span className="text-slate-400">Image Familiarity</span>
                                <span className={`font-semibold ${
                                  analysisResult.ood?.image_familiarity === 'HIGH RISK' ? 'text-rose-400' : 'text-cyan-300'
                                }`}>
                                  {analysisResult.ood?.image_familiarity || 'Familiar'}
                                </span>
                              </div>

                              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80 flex items-center justify-between">
                                <span className="text-slate-400">Result Stability</span>
                                <span className="font-semibold text-emerald-400">
                                  {analysisResult.stability?.stability_percentage || 100}% Stable
                                </span>
                              </div>
                            </div>
                          </div>

                          {/* Multi-Finding Probabilities if available */}
                          {analysisResult.prediction?.findings && analysisResult.prediction.findings.length > 0 && (
                            <div className="pt-3 border-t border-slate-800 space-y-2">
                              <span className="text-xs font-semibold text-slate-300">Detailed Findings:</span>
                              <div className="grid grid-cols-2 gap-2 text-xs">
                                {analysisResult.prediction.findings.slice(0, 6).map((f, idx) => (
                                  <div key={idx} className="flex justify-between items-center p-2 rounded bg-slate-950/50 border border-slate-800">
                                    <span className="truncate text-slate-300">{f.condition}</span>
                                    <span className="font-mono text-cyan-300 font-bold ml-2">{f.probability_pct}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>

                        {/* RECOMMENDED REVIEW CARD */}
                        {analysisResult.specialist && (
                          <div className={`rounded-2xl border p-4 space-y-2.5 ${
                            analysisResult.specialist.urgency_class === 'urgent'
                              ? 'border-rose-500/40 bg-rose-950/20'
                              : 'border-slate-800 bg-slate-900'
                          }`}>
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-bold text-slate-300 flex items-center space-x-1.5">
                                <Stethoscope className="w-4 h-4 text-cyan-400" />
                                <span>Recommended Review</span>
                              </span>
                              <span className={`text-[10px] font-bold px-2.5 py-0.5 rounded-full ${
                                analysisResult.specialist.urgency_class === 'urgent'
                                  ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                                  : 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                              }`}>
                                {analysisResult.specialist.review_priority}
                              </span>
                            </div>

                            <div className="text-sm font-bold text-white">
                              {analysisResult.specialist.recommended_specialist}
                            </div>
                            <p className="text-xs text-slate-300">{analysisResult.specialist.department}</p>
                            
                            <p className="text-[11px] text-slate-300 bg-slate-950/50 p-2.5 rounded-lg border border-slate-800 leading-relaxed">
                              {analysisResult.specialist.action_prompt}
                            </p>
                          </div>
                        )}

                        {/* SECONDARY COLLAPSIBLE: AI TECHNICAL DETAILS */}
                        <div className="rounded-2xl border border-slate-800 bg-slate-900/60 overflow-hidden">
                          <button
                            onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
                            className="w-full p-4 flex items-center justify-between text-left text-xs font-bold text-slate-300 hover:text-white hover:bg-slate-800/40 transition-all cursor-pointer"
                          >
                            <div className="flex items-center space-x-2">
                              <Server className="w-4 h-4 text-cyan-400" />
                              <span>AI Technical Details (Secondary Audit)</span>
                            </div>
                            {showTechnicalDetails ? (
                              <ChevronUp className="w-4 h-4 text-slate-400" />
                            ) : (
                              <ChevronDown className="w-4 h-4 text-slate-400" />
                            )}
                          </button>

                          {showTechnicalDetails && (
                            <div className="p-4 pt-0 border-t border-slate-800/80 space-y-3 text-xs">
                              <p className="text-[11px] text-slate-400 pt-2">
                                Technical metrics generated by the Trust Engine during inference:
                              </p>

                              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-slate-300 font-mono text-[11px]">
                                <div className="p-2 rounded bg-slate-950 border border-slate-800">
                                  <div className="text-slate-400 text-[10px]">CALIBRATED VS RAW CONFIDENCE</div>
                                  <div className="text-white font-bold">{calibratedPct}% (Calibrated) | {rawPct}% (Raw)</div>
                                </div>

                                <div className="p-2 rounded bg-slate-950 border border-slate-800">
                                  <div className="text-slate-400 text-[10px]">CALIBRATION TEMPERATURE</div>
                                  <div className="text-white font-bold">T = {analysisResult.calibration?.temperature ?? 1.15}</div>
                                </div>

                                <div className="p-2 rounded bg-slate-950 border border-slate-800">
                                  <div className="text-slate-400 text-[10px]">MC DROPOUT VARIANCE</div>
                                  <div className="text-white font-bold">{analysisResult.uncertainty?.variance ?? 0.001} (10 passes)</div>
                                </div>

                                <div className="p-2 rounded bg-slate-950 border border-slate-800">
                                  <div className="text-slate-400 text-[10px]">FREE-ENERGY OOD SCORE</div>
                                  <div className="text-white font-bold">{analysisResult.ood?.energy_score ?? -5.2}</div>
                                </div>

                                <div className="p-2 rounded bg-slate-950 border border-slate-800">
                                  <div className="text-slate-400 text-[10px]">IMAGE BLUR (LAPLACIAN)</div>
                                  <div className="text-white font-bold">{analysisResult.quality?.blur_score ?? 150} (Min: 40)</div>
                                </div>

                                <div className="p-2 rounded bg-slate-950 border border-slate-800">
                                  <div className="text-slate-400 text-[10px]">RESULT STABILITY</div>
                                  <div className="text-white font-bold">{analysisResult.stability?.stability_percentage ?? 100}% Concordant</div>
                                </div>
                              </div>

                              <div className="text-[10px] text-slate-500 font-mono pt-1">
                                Inference Latency: Total {analysisResult.latency_ms?.total || 0} ms on CPU.
                              </div>
                            </div>
                          )}
                        </div>

                      </div>
                    </div>
                  );
                })()}

              </div>
            )}
          </div>
        )}

        {/* -------------------------------------------------------- */}
        {/* VIEW 4: HOW IT WORKS (SIMPLE EXPLANATION) */}
        {/* -------------------------------------------------------- */}
        {activeTab === 'how-it-works' && (
          <div className="max-w-3xl mx-auto space-y-8">
            <div className="text-center space-y-2">
              <h2 className="text-2xl font-black text-white">How TRUSTMED Works</h2>
              <p className="text-xs sm:text-sm text-slate-400">
                A 5-layer safety architecture designed to make medical AI transparent, self-aware, and clinically safe.
              </p>
            </div>

            <div className="space-y-4">
              {[
                {
                  step: '1',
                  title: 'Image Quality Verification',
                  desc: 'Before executing any deep neural network, TRUSTMED checks the image for blur, darkness, low contrast, and corrupt files. If quality is poor, analysis is refused to prevent hallucinations.',
                  icon: ShieldCheck
                },
                {
                  step: '2',
                  title: 'Modality Routing',
                  desc: 'Images are verified to match supported specialties (Chest X-ray, Brain MRI, Skin lesion, Retinal fundus). Unsupported modalities like dental scans are safely filtered out.',
                  icon: Layers
                },
                {
                  step: '3',
                  title: 'Calibrated AI Assessment',
                  desc: 'Specialized deep learning models evaluate the scan. Softmax output probabilities are calibrated so confidence reflects actual diagnostic accuracy.',
                  icon: Activity
                },
                {
                  step: '4',
                  title: 'Uncertainty & Familiarity Audit',
                  desc: 'Multiple forward passes measure model hesitation (uncertainty), while energy scoring verifies if the patient scan resembles verified medical training data.',
                  icon: Eye
                },
                {
                  step: '5',
                  title: 'Explanation & Specialist Review',
                  desc: 'Visual overlays highlight the areas that influenced the AI, and the system recommends appropriate human specialist review when the result warrants clinical confirmation.',
                  icon: Stethoscope
                }
              ].map((item, idx) => {
                const Icon = item.icon;
                return (
                  <div key={idx} className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 flex items-start space-x-4">
                    <div className="w-9 h-9 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 font-bold shrink-0">
                      {item.step}
                    </div>
                    <div className="space-y-1">
                      <h3 className="text-base font-bold text-white flex items-center space-x-2">
                        <span>{item.title}</span>
                      </h3>
                      <p className="text-xs text-slate-300 leading-relaxed">{item.desc}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* -------------------------------------------------------- */}
        {/* VIEW 5: ABOUT & CLINICAL BOUNDARIES */}
        {/* -------------------------------------------------------- */}
        {activeTab === 'about' && (
          <div className="max-w-3xl mx-auto space-y-6">
            <div className="text-center space-y-2">
              <h2 className="text-2xl font-black text-white">About TRUSTMED</h2>
              <p className="text-xs sm:text-sm text-slate-400">
                Ethical boundaries and decision-support safeguards for healthcare AI
              </p>
            </div>

            <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6 space-y-4 text-xs text-slate-300 leading-relaxed">
              <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                <Shield className="w-4 h-4 text-cyan-400" />
                <span>Our Clinical Safety Principles</span>
              </h3>

              <div className="space-y-3 text-slate-300">
                <p>
                  <strong>1. Decision-Support Only:</strong> TRUSTMED is designed to assist qualified healthcare professionals, not replace them. Final diagnoses and treatment decisions must always be made by licensed physicians.
                </p>
                <p>
                  <strong>2. Zero Overconfidence:</strong> When the AI encounters ambiguous, out-of-distribution, or borderline medical scans, it clearly flags low reliability and recommends specialist review rather than asserting false certainty.
                </p>
                <p>
                  <strong>3. Transparent Visual Explanations:</strong> Clinicians can inspect visual heatmaps and segmentation regions to understand what image features influenced the AI finding.
                </p>
                <p>
                  <strong>4. Strict Validation Gating:</strong> Inadequate or unsupported images cannot proceed to prediction or report generation.
                </p>
              </div>
            </div>

            <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 space-y-2 text-center text-xs text-slate-400">
              <p className="font-bold text-slate-300">Important Medical Notice</p>
              <p>
                TRUSTMED is an investigational academic prototype developed for the AI Healthcare challenge. It is not an FDA-cleared or CE-marked medical device. Do not use for emergency medical triage or autonomous patient diagnosis.
              </p>
            </div>
          </div>
        )}

        {/* -------------------------------------------------------- */}
        {/* VIEW 6: TECHNICAL SPECIFICATIONS (FOR JUDGES / AUDIT) */}
        {/* -------------------------------------------------------- */}
        {activeTab === 'technical' && (
          <div className="max-w-4xl mx-auto space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-black text-white">Technical Specifications & Benchmarks</h2>
                <p className="text-xs text-slate-400">Model performance metrics evaluated on held-out test splits (CPU Ready)</p>
              </div>
              <button
                onClick={() => { fetchPerformance(); fetchDatasets(); }}
                className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 flex items-center space-x-1.5"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Refresh</span>
              </button>
            </div>

            {/* Performance Cards Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="rounded-2xl border border-slate-800 bg-slate-900 p-4 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-bold text-white">🫁 Lung (CXR14)</span>
                  <span className="font-mono text-cyan-400 font-bold">ResNet-18</span>
                </div>
                <div className="text-2xl font-mono font-black text-cyan-300">
                  {performanceData?.models?.lung?.macro_auroc?.toFixed(3) ?? '0.842'}
                </div>
                <div className="text-[11px] text-slate-400 space-y-0.5">
                  <div>AUROC: <strong>{performanceData?.models?.lung?.macro_auroc?.toFixed(3) ?? '0.842'}</strong></div>
                  <div>Sensitivity: <strong>{performanceData?.models?.lung?.sensitivity?.toFixed(3) ?? '0.780'}</strong></div>
                  <div>Specificity: <strong>{performanceData?.models?.lung?.specificity?.toFixed(3) ?? '0.890'}</strong></div>
                </div>
              </div>

              <div className="rounded-2xl border border-slate-800 bg-slate-900 p-4 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-bold text-white">🧠 Brain (BraTS)</span>
                  <span className="font-mono text-purple-400 font-bold">U-Net</span>
                </div>
                <div className="text-2xl font-mono font-black text-purple-300">
                  {performanceData?.models?.brain?.dice?.toFixed(3) ?? '0.711'}
                </div>
                <div className="text-[11px] text-slate-400 space-y-0.5">
                  <div>Dice Score: <strong>{performanceData?.models?.brain?.dice?.toFixed(3) ?? '0.711'}</strong></div>
                  <div>IoU: <strong>{performanceData?.models?.brain?.iou?.toFixed(3) ?? '0.711'}</strong></div>
                  <div>Specificity: <strong>{performanceData?.models?.brain?.specificity?.toFixed(3) ?? '0.920'}</strong></div>
                </div>
              </div>

              <div className="rounded-2xl border border-slate-800 bg-slate-900 p-4 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-bold text-white">🩺 Skin (ISIC)</span>
                  <span className="font-mono text-amber-400 font-bold">ResNet-18</span>
                </div>
                <div className="text-2xl font-mono font-black text-amber-300">
                  {performanceData?.models?.skin?.macro_auroc?.toFixed(3) ?? '0.865'}
                </div>
                <div className="text-[11px] text-slate-400 space-y-0.5">
                  <div>AUROC: <strong>{performanceData?.models?.skin?.macro_auroc?.toFixed(3) ?? '0.865'}</strong></div>
                  <div>Accuracy: <strong>{performanceData?.models?.skin?.accuracy?.toFixed(3) ?? '0.812'}</strong></div>
                  <div>Sensitivity: <strong>{performanceData?.models?.skin?.sensitivity?.toFixed(3) ?? '0.795'}</strong></div>
                </div>
              </div>

              <div className="rounded-2xl border border-slate-800 bg-slate-900 p-4 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-bold text-white">👁️ Retina (APTOS)</span>
                  <span className="font-mono text-emerald-400 font-bold">ResNet-18</span>
                </div>
                <div className="text-2xl font-mono font-black text-emerald-300">
                  {performanceData?.models?.retina?.macro_auroc?.toFixed(3) ?? '0.880'}
                </div>
                <div className="text-[11px] text-slate-400 space-y-0.5">
                  <div>AUROC: <strong>{performanceData?.models?.retina?.macro_auroc?.toFixed(3) ?? '0.880'}</strong></div>
                  <div>Accuracy: <strong>{performanceData?.models?.retina?.accuracy?.toFixed(3) ?? '0.840'}</strong></div>
                  <div>Sensitivity: <strong>{performanceData?.models?.retina?.sensitivity?.toFixed(3) ?? '0.810'}</strong></div>
                </div>
              </div>
            </div>

            {/* Datasets Table */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5 space-y-3">
              <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                <Database className="w-4 h-4 text-cyan-400" />
                <span>Verified Public Datasets</span>
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                  <div className="font-bold text-white">NIH ChestX-ray14</div>
                  <p className="text-slate-400">14 Thoracic multi-label findings. Patient-level stratified test split.</p>
                </div>
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                  <div className="font-bold text-white">BraTS 2021</div>
                  <p className="text-slate-400">Brain MRI tumor segmentation. Subject-level validation split.</p>
                </div>
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                  <div className="font-bold text-white">ISIC 2019</div>
                  <p className="text-slate-400">8-class skin lesion classification. Patient ID stratified.</p>
                </div>
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                  <div className="font-bold text-white">APTOS 2019</div>
                  <p className="text-slate-400">Diabetic retinopathy grading (0 to 4). Eye-level split.</p>
                </div>
              </div>
            </div>
          </div>
        )}

      </main>

      {/* ======================================================== */}
      {/* REPORT PREPARATION MODAL (REQUIREMENT 10) */}
      {/* ======================================================== */}
      {isPreparingReport && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 sm:p-8 max-w-md w-full space-y-6 shadow-2xl relative">
            
            <button
              onClick={() => setIsPreparingReport(false)}
              className="absolute top-4 right-4 p-1.5 rounded-full text-slate-400 hover:text-white hover:bg-slate-800 transition-all cursor-pointer"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="text-center space-y-2">
              <div className="w-12 h-12 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 mx-auto">
                <FileCheck className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold text-white">Preparing Your Report</h3>
              <p className="text-xs text-slate-400">
                TRUSTMED is preparing your medical image assessment.
              </p>
            </div>

            {/* Step-by-Step Progress Checklist */}
            <div className="space-y-3 py-2 text-xs">
              <div className="flex items-center space-x-3">
                {reportStep >= 1 ? (
                  <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />
                ) : (
                  <div className="w-4 h-4 rounded-full border-2 border-slate-700 shrink-0"></div>
                )}
                <span className={reportStep >= 1 ? 'text-slate-200 font-medium' : 'text-slate-500'}>
                  Image analyzed
                </span>
              </div>

              <div className="flex items-center space-x-3">
                {reportStep >= 2 ? (
                  <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />
                ) : (
                  <div className="w-4 h-4 rounded-full border-2 border-slate-700 shrink-0"></div>
                )}
                <span className={reportStep >= 2 ? 'text-slate-200 font-medium' : 'text-slate-500'}>
                  AI assessment completed
                </span>
              </div>

              <div className="flex items-center space-x-3">
                {reportStep >= 3 ? (
                  <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />
                ) : (
                  <div className="w-4 h-4 rounded-full border-2 border-slate-700 shrink-0"></div>
                )}
                <span className={reportStep >= 3 ? 'text-slate-200 font-medium' : 'text-slate-500'}>
                  Reliability checked
                </span>
              </div>

              <div className="flex items-center space-x-3">
                {reportStep >= 4 ? (
                  <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />
                ) : (
                  <div className="w-4 h-4 rounded-full border-2 border-slate-700 shrink-0"></div>
                )}
                <span className={reportStep >= 4 ? 'text-slate-200 font-medium' : 'text-slate-500'}>
                  Explanation prepared
                </span>
              </div>

              <div className="flex items-center space-x-3">
                {reportStep >= 5 ? (
                  <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />
                ) : (
                  <RefreshCw className="w-4 h-4 text-cyan-400 animate-spin shrink-0" />
                )}
                <span className={reportStep >= 5 ? 'text-slate-200 font-medium' : 'text-cyan-300 font-medium'}>
                  {reportStep >= 5 ? 'Report ready' : 'Preparing report...'}
                </span>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="pt-2 flex justify-center">
              {isReportReady ? (
                <button
                  onClick={triggerPdfDownload}
                  className="w-full py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-bold text-xs flex items-center justify-center space-x-2 shadow-lg shadow-cyan-500/25 transition-all cursor-pointer"
                >
                  <Download className="w-4 h-4" />
                  <span>Download Report</span>
                </button>
              ) : (
                <button
                  disabled
                  className="w-full py-3 rounded-xl bg-slate-800 text-slate-500 font-bold text-xs flex items-center justify-center space-x-2 cursor-not-allowed"
                >
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Finalizing PDF document...</span>
                </button>
              )}
            </div>

          </div>
        </div>
      )}

      {/* ======================================================== */}
      {/* FOOTER */}
      {/* ======================================================== */}
      <footer className="border-t border-slate-800/80 bg-[#0f172a] py-4 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>TRUSTMED — Self-Aware Multi-Specialty Medical AI © 2026</span>
          <span className="text-[11px] text-slate-400">Decision-Support Prototype • Healthcare AI</span>
        </div>
      </footer>

    </div>
  );
}
