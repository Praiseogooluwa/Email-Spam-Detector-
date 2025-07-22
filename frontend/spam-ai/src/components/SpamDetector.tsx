import React, { useState, useRef, useEffect } from 'react';
import { 
  Mail, Shield, AlertTriangle, CheckCircle, Upload, Send, Bot, Globe, User, 
  FileText, Zap, Brain, Eye, AlertCircle, Loader2, X, Moon, Sun, Sparkles,
  Lock, Scan, TrendingUp, BarChart3
} from 'lucide-react';
import { useTheme } from '@/components/theme-provider';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { useToast } from '@/hooks/use-toast';

const SpamDetector = () => {
  const [emailText, setEmailText] = useState('');
  const [senderEmail, setSenderEmail] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('text');
  const [dragOver, setDragOver] = useState(false);
  const [mounted, setMounted] = useState(false);
  const fileInputRef = useRef(null);
  const { theme, setTheme } = useTheme();
  const { toast } = useToast();

  useEffect(() => {
    setMounted(true);
  }, []);

  const API_BASE_URL = 'http://127.0.0.1:8000';

  const analyzeEmail = async () => {
    if (!emailText.trim() || !senderEmail.trim()) {
      toast({
        title: "Missing Information",
        description: "Please fill in both email content and sender address",
        variant: "destructive",
      });
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: emailText,
          sender: senderEmail,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
      
      toast({
        title: "Analysis Complete",
        description: `Email classified as ${data.label}`,
        variant: data.label === 'SPAM' ? "destructive" : "default",
      });
    } catch (err) {
      const errorMessage = `Failed to analyze email: ${err.message}`;
      setError(errorMessage);
      toast({
        title: "Analysis Failed",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (file) => {
    if (!file || !file.name.endsWith('.eml')) {
      const errorMessage = 'Please upload a valid .eml file';
      setError(errorMessage);
      toast({
        title: "Invalid File",
        description: errorMessage,
        variant: "destructive",
      });
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_BASE_URL}/upload_eml`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
      
      if (data.file_info?.extracted_sender) {
        setSenderEmail(data.file_info.extracted_sender);
      }

      toast({
        title: "File Uploaded Successfully",
        description: `Analysis complete for ${file.name}`,
      });
    } catch (err) {
      const errorMessage = `Failed to upload file: ${err.message}`;
      setError(errorMessage);
      toast({
        title: "Upload Failed",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const clearResults = () => {
    setResult(null);
    setError('');
  };

  const getRiskVariant = (risk) => {
    switch (risk) {
      case 'High Risk': return 'destructive';
      case 'Low Risk': return 'default';
      default: return 'secondary';
    }
  };

  const getBotVariant = (bot) => {
    return bot === 'Likely Bot-Written' ? 'outline' : 'secondary';
  };

  const getSpamVariant = (label) => {
    return label === 'SPAM' ? 'destructive' : 'default';
  };

  const sampleEmails = [
    {
      name: "Phishing Example",
      text: "URGENT: Your PayPal account has been limited. Click here to verify: http://fake-paypal.com/verify",
      sender: "security@paypal-verify.tk",
      icon: AlertTriangle,
      color: "text-red-500"
    },
    {
      name: "Legitimate Email",
      text: "Hi John, can you please send me the quarterly sales report by end of day Friday? Thanks!",
      sender: "sarah@company.com",
      icon: CheckCircle,
      color: "text-green-500"
    },
    {
      name: "Lottery Scam",
      text: "CONGRATULATIONS! You've won £850,000 in the UK National Lottery! Claim code: UK2024WIN",
      sender: "winner@uk-lottery.ru",
      icon: TrendingUp,
      color: "text-orange-500"
    }
  ];

  const toggleTheme = () => {
    setTheme(theme === "light" ? "dark" : "light");
  };

  if (!mounted) {
    return null; // Avoid hydration mismatch
  }

  return (
    <div className="min-h-screen bg-background transition-all duration-500">
      {/* Animated Background */}
      <div className="fixed inset-0 opacity-30 pointer-events-none">
        <div className="absolute top-20 left-20 w-72 h-72 bg-primary/20 rounded-full mix-blend-multiply filter blur-xl animate-float"></div>
        <div className="absolute top-40 right-20 w-72 h-72 bg-blue-400/20 rounded-full mix-blend-multiply filter blur-xl animate-float" style={{animationDelay: '2s'}}></div>
        <div className="absolute -bottom-8 left-40 w-72 h-72 bg-purple-400/20 rounded-full mix-blend-multiply filter blur-xl animate-float" style={{animationDelay: '4s'}}></div>
      </div>

      {/* Header */}
      <header className="relative border-b border-border bg-glass backdrop-blur-xl">
      <div className="container mx-auto px-4 sm:px-6 py-4 sm:py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 sm:space-x-4 animate-fade-in">
              <div className="relative">
                <div className="p-2 sm:p-3 bg-hero-gradient rounded-xl sm:rounded-2xl shadow-glow animate-pulse-glow">
                  <Shield className="h-6 w-6 sm:h-8 sm:w-8 text-white" />
                </div>
                <div className="absolute -top-1 -right-1 h-4 w-4 sm:h-6 sm:w-6 bg-hero-gradient rounded-full flex items-center justify-center animate-bounce">
                  <Brain className="h-2 w-2 sm:h-3 sm:w-3 text-white" />
                </div>
              </div>
              <div className="min-w-0">
                <h1 className="text-xl sm:text-2xl lg:text-3xl font-bold bg-hero-gradient bg-clip-text text-transparent truncate">
                  AI Spam Guardian
                </h1>
                <p className="text-xs sm:text-sm text-muted-foreground hidden sm:block">Advanced email security powered by artificial intelligence</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-2 sm:space-x-4 animate-slide-in-right">
              <Button
                variant="outline"
                size="icon"
                onClick={toggleTheme}
                className="relative overflow-hidden group border-glass h-8 w-8 sm:h-10 sm:w-10"
              >
                <Sun className="h-4 w-4 sm:h-[1.2rem] sm:w-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
                <Moon className="absolute h-4 w-4 sm:h-[1.2rem] sm:w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
                <span className="sr-only">Toggle theme</span>
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 sm:px-6 py-6 sm:py-8">
        <div className="max-w-7xl mx-auto">
          {/* Navigation Tabs */}
          <div className="flex justify-center mb-6 sm:mb-8 animate-fade-in-up">
            <Card className="p-1 bg-glass border-glass w-full max-w-md">
              <div className="flex">
                <Button
                  onClick={() => setActiveTab('text')}
                  variant={activeTab === 'text' ? 'default' : 'ghost'}
                  className={`flex-1 flex items-center justify-center space-x-1 sm:space-x-2 transition-all duration-300 text-xs sm:text-sm ${
                    activeTab === 'text' 
                      ? 'bg-hero-gradient text-white shadow-glow' 
                      : 'hover:bg-muted'
                  }`}
                >
                  <FileText className="h-3 w-3 sm:h-4 sm:w-4" />
                  <span className="hidden sm:inline">Text Analysis</span>
                  <span className="sm:hidden">Text</span>
                </Button>
                <Button
                  onClick={() => setActiveTab('upload')}
                  variant={activeTab === 'upload' ? 'default' : 'ghost'}
                  className={`flex-1 flex items-center justify-center space-x-1 sm:space-x-2 transition-all duration-300 text-xs sm:text-sm ${
                    activeTab === 'upload' 
                      ? 'bg-hero-gradient text-white shadow-glow' 
                      : 'hover:bg-muted'
                  }`}
                >
                  <Upload className="h-3 w-3 sm:h-4 sm:w-4" />
                  <span className="hidden sm:inline">Upload .EML</span>
                  <span className="sm:hidden">Upload</span>
                </Button>
              </div>
            </Card>
          </div>

          <div className="grid lg:grid-cols-2 gap-6 lg:gap-8">
            {/* Left Panel - Input */}
            <div className="space-y-4 sm:space-y-6 animate-fade-in-up" style={{animationDelay: '0.1s'}}>
              {activeTab === 'text' ? (
                <Card className="bg-glass border-glass backdrop-blur-xl shadow-glow">
                  <CardHeader className="pb-4">
                    <CardTitle className="flex items-center space-x-2 sm:space-x-3 text-base sm:text-lg">
                      <div className="p-1.5 sm:p-2 bg-hero-gradient rounded-lg">
                        <Mail className="h-4 w-4 sm:h-5 sm:w-5 text-white" />
                      </div>
                      <span>Email Analysis</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4 pt-0">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Sender Email Address</label>
                      <Input
                        type="email"
                        value={senderEmail}
                        onChange={(e) => setSenderEmail(e.target.value)}
                        placeholder="sender@example.com"
                        className="transition-all duration-300 focus:shadow-glow text-sm"
                      />
                    </div>
                    
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Email Content</label>
                      <Textarea
                        value={emailText}
                        onChange={(e) => setEmailText(e.target.value)}
                        placeholder="Paste your email content here..."
                        rows={6}
                        className="transition-all duration-300 focus:shadow-glow resize-none text-sm"
                      />
                    </div>
                    
                    <Button
                      onClick={analyzeEmail}
                      disabled={loading}
                      className="w-full bg-hero-gradient hover:bg-hero-gradient-hover text-white shadow-glow hover:shadow-xl transition-all duration-300 group h-10 sm:h-11"
                    >
                      {loading ? (
                        <>
                          <Loader2 className="h-4 w-4 animate-spin mr-2" />
                          <span className="text-sm">Analyzing...</span>
                        </>
                      ) : (
                        <>
                          <Scan className="h-4 w-4 mr-2 group-hover:animate-pulse" />
                          <span className="text-sm">Analyze Email</span>
                          <Sparkles className="h-4 w-4 ml-2" />
                        </>
                      )}
                    </Button>
                  </CardContent>
                </Card>
              ) : (
                <Card className="bg-glass border-glass backdrop-blur-xl shadow-glow">
                  <CardHeader className="pb-4">
                    <CardTitle className="flex items-center space-x-2 sm:space-x-3 text-base sm:text-lg">
                      <div className="p-1.5 sm:p-2 bg-hero-gradient rounded-lg">
                        <Upload className="h-4 w-4 sm:h-5 sm:w-5 text-white" />
                      </div>
                      <span>Upload Email File</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <div
                      className={`border-2 border-dashed rounded-xl p-6 sm:p-8 text-center transition-all duration-300 cursor-pointer group ${
                        dragOver
                          ? 'border-primary bg-primary/10 scale-105'
                          : 'border-border hover:border-primary/50 hover:bg-muted/50'
                      }`}
                      onDrop={handleDrop}
                      onDragOver={handleDragOver}
                      onDragLeave={handleDragLeave}
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <Upload className={`h-8 w-8 sm:h-12 sm:w-12 mx-auto mb-3 sm:mb-4 transition-all duration-300 ${
                        dragOver ? 'text-primary animate-bounce' : 'text-muted-foreground group-hover:text-primary'
                      }`} />
                      <p className="text-sm sm:text-base text-foreground mb-2">
                        <span className="hidden sm:inline">Drag and drop your .eml file here, or </span>
                        <span className="text-primary font-medium hover:underline">
                          <span className="sm:hidden">Tap to </span>browse
                        </span>
                      </p>
                      <p className="text-xs sm:text-sm text-muted-foreground">Supports .eml email files</p>
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept=".eml"
                        onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
                        className="hidden"
                      />
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Sample Emails */}
              <Card className="bg-glass border-glass backdrop-blur-xl shadow-glow">
                <CardHeader className="pb-4">
                  <CardTitle className="flex items-center space-x-2 text-base sm:text-lg">
                    <Eye className="h-4 w-4 sm:h-5 sm:w-5 text-primary" />
                    <span>Try Sample Emails</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 sm:space-y-3 pt-0">
                  {sampleEmails.map((sample, index) => {
                    const IconComponent = sample.icon;
                    return (
                      <button
                        key={index}
                        onClick={() => {
                          setEmailText(sample.text);
                          setSenderEmail(sample.sender);
                          setActiveTab('text');
                        }}
                        className="w-full text-left p-3 sm:p-4 bg-muted/50 hover:bg-muted rounded-lg border border-border hover:border-primary/50 transition-all duration-300 group hover:shadow-md"
                      >
                        <div className="flex items-start space-x-2 sm:space-x-3">
                          <IconComponent className={`h-4 w-4 sm:h-5 sm:w-5 mt-0.5 flex-shrink-0 ${sample.color} group-hover:animate-pulse`} />
                          <div className="flex-1 min-w-0">
                            <p className="text-sm sm:text-base font-medium text-foreground group-hover:text-primary transition-colors">
                              {sample.name}
                            </p>
                            <p className="text-xs sm:text-sm text-muted-foreground mt-1 line-clamp-2">
                              {sample.text}
                            </p>
                          </div>
                        </div>
                      </button>
                    );
                  })}
                </CardContent>
              </Card>
            </div>

            {/* Right Panel - Results */}
            <div className="space-y-4 sm:space-y-6 animate-fade-in-up" style={{animationDelay: '0.2s'}}>
              {error && (
                <Card className="border-destructive bg-destructive/5 animate-scale-in">
                  <CardContent className="pt-4 sm:pt-6">
                    <div className="flex items-start space-x-2 sm:space-x-3">
                      <AlertCircle className="h-4 w-4 sm:h-5 sm:w-5 text-destructive mt-0.5 flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <h3 className="text-destructive font-medium text-sm sm:text-base">Error</h3>
                        <p className="text-destructive/80 text-xs sm:text-sm mt-1 break-words">{error}</p>
                      </div>
                      <Button
                        onClick={() => setError('')}
                        variant="ghost"
                        size="sm"
                        className="text-destructive hover:text-destructive/80 h-6 w-6 sm:h-8 sm:w-8 p-0"
                      >
                        <X className="h-3 w-3 sm:h-4 sm:w-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )}

              {result ? (
                <Card className="bg-glass border-glass backdrop-blur-xl shadow-glow animate-scale-in">
                  <CardHeader className="border-b border-border">
                    <div className="flex items-center justify-between">
                      <CardTitle className="flex items-center space-x-2">
                        <Brain className="h-5 w-5 text-primary animate-pulse" />
                        <span>Analysis Results</span>
                      </CardTitle>
                      <Button
                        onClick={clearResults}
                        variant="ghost"
                        size="sm"
                        className="hover:bg-destructive/10 hover:text-destructive"
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardHeader>

                  <CardContent className="pt-6 space-y-6">
                    {/* Main Result */}
                    <div className={`p-6 rounded-xl border-2 ${
                      result.label === 'SPAM' 
                        ? 'border-destructive bg-destructive/5' 
                        : 'border-green-500 bg-green-500/5'
                    } animate-scale-in shadow-lg`}>
                      <div className="flex items-center space-x-3">
                        {result.label === 'SPAM' ? (
                          <AlertTriangle className="h-8 w-8 text-destructive animate-pulse" />
                        ) : (
                          <CheckCircle className="h-8 w-8 text-green-500 animate-bounce" />
                        )}
                        <div className="flex-1">
                          <p className="text-xl font-bold">
                            {result.label === 'SPAM' ? 'SPAM DETECTED' : 'LEGITIMATE EMAIL'}
                          </p>
                          <div className="flex items-center space-x-2 mt-1">
                            <BarChart3 className="h-4 w-4" />
                            <span className="text-sm font-medium">
                              Confidence: {result.confidence}%
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Details Grid */}
                    <div className="grid grid-cols-2 gap-3 sm:gap-4">
                      <Card className="p-3 sm:p-4 hover:shadow-glow transition-all duration-300">
                        <div className="flex items-center space-x-1 sm:space-x-2 mb-2">
                          <User className="h-3 w-3 sm:h-4 sm:w-4 text-primary flex-shrink-0" />
                          <span className="font-medium text-xs sm:text-sm truncate">Sender Risk</span>
                        </div>
                        <Badge variant={getRiskVariant(result.sender_risk)} className="text-xs truncate w-full justify-center">
                          {result.sender_risk}
                        </Badge>
                      </Card>

                      <Card className="p-3 sm:p-4 hover:shadow-glow transition-all duration-300">
                        <div className="flex items-center space-x-1 sm:space-x-2 mb-2">
                          <Bot className="h-3 w-3 sm:h-4 sm:w-4 text-primary flex-shrink-0" />
                          <span className="font-medium text-xs sm:text-sm truncate">Content</span>
                        </div>
                        <Badge variant={getBotVariant(result.bot_likeness)} className="text-xs truncate w-full justify-center">
                          {result.bot_likeness.split('-')[0]}
                        </Badge>
                      </Card>

                      <Card className="p-3 sm:p-4 hover:shadow-glow transition-all duration-300">
                        <div className="flex items-center space-x-1 sm:space-x-2 mb-2">
                          <Globe className="h-3 w-3 sm:h-4 sm:w-4 text-primary flex-shrink-0" />
                          <span className="font-medium text-xs sm:text-sm truncate">Language</span>
                        </div>
                        <Badge variant="outline" className="text-xs uppercase truncate w-full justify-center">
                          {result.language}
                        </Badge>
                      </Card>

                      <Card className="p-3 sm:p-4 hover:shadow-glow transition-all duration-300">
                        <div className="flex items-center space-x-1 sm:space-x-2 mb-2">
                          <Lock className="h-3 w-3 sm:h-4 sm:w-4 text-primary flex-shrink-0" />
                          <span className="font-medium text-xs sm:text-sm truncate">Keywords</span>
                        </div>
                        <Badge variant="secondary" className="text-xs truncate w-full justify-center">
                          {result.spammy_keywords.length} found
                        </Badge>
                      </Card>
                    </div>

                    {/* Spam Keywords */}
                    {result.spammy_keywords.length > 0 && (
                      <Card className="p-4 bg-warning-subtle border-warning-subtle">
                        <h4 className="font-medium mb-3 flex items-center space-x-2">
                          <AlertTriangle className="h-4 w-4 text-warning" />
                          <span>Suspicious Keywords Found</span>
                        </h4>
                        <div className="flex flex-wrap gap-2">
                          {result.spammy_keywords.map((keyword, index) => (
                            <Badge
                              key={index}
                              variant="destructive"
                              className="text-xs animate-fade-in"
                              style={{animationDelay: `${index * 0.1}s`}}
                            >
                              {keyword}
                            </Badge>
                          ))}
                        </div>
                      </Card>
                    )}

                    {/* Explanation */}
                    <Card className="p-4 bg-muted/50">
                      <h4 className="font-medium mb-3 flex items-center space-x-2">
                        <Brain className="h-4 w-4 text-primary animate-pulse" />
                        <span>AI Explanation</span>
                      </h4>
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        {result.explanation}
                      </p>
                    </Card>

                    {/* File Info (if uploaded) */}
                    {result.file_info && (
                      <Card className="p-4 bg-muted/50">
                        <h4 className="font-medium mb-3 flex items-center space-x-2">
                          <FileText className="h-4 w-4 text-primary" />
                          <span>File Information</span>
                        </h4>
                        <div className="text-sm space-y-1">
                          <p><span className="text-muted-foreground">Filename:</span> <span className="font-mono">{result.file_info.filename}</span></p>
                          <p><span className="text-muted-foreground">Size:</span> {(result.file_info.size_bytes / 1024).toFixed(1)} KB</p>
                          <p><span className="text-muted-foreground">Extracted Sender:</span> <span className="font-mono">{result.file_info.extracted_sender}</span></p>
                        </div>
                      </Card>
                    )}
                  </CardContent>
                </Card>
              ) : (
                <Card className="bg-glass border-glass backdrop-blur-xl shadow-glow text-center p-6 sm:p-12 animate-fade-in">
                  <CardContent className="pt-4 sm:pt-6">
                    <div className="w-16 h-16 sm:w-20 sm:h-20 bg-hero-gradient rounded-full flex items-center justify-center mx-auto mb-4 sm:mb-6 animate-float">
                      <Shield className="h-8 w-8 sm:h-10 sm:w-10 text-white" />
                    </div>
                    <h3 className="text-lg sm:text-2xl font-bold mb-2 sm:mb-3 bg-hero-gradient bg-clip-text text-transparent">
                      Ready to Analyze
                    </h3>
                    <p className="text-sm sm:text-base text-muted-foreground max-w-md mx-auto">
                      Enter email content or upload an .eml file to get started with our advanced AI-powered spam detection
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border bg-glass backdrop-blur-xl mt-auto">
        <div className="container mx-auto px-4 sm:px-6 py-4 sm:py-6">
          <div className="flex flex-col sm:flex-row items-center justify-center space-y-1 sm:space-y-0 sm:space-x-2 text-xs sm:text-sm text-muted-foreground text-center">
            <div className="flex items-center space-x-2">
              <Shield className="h-3 w-3 sm:h-4 sm:w-4 text-primary" />
              <span>AI Spam Guardian by Praise Ogooluwa</span>
            </div>
            <span className="hidden sm:inline">•</span>
            <span className="text-xs">Advanced email security with machine learning</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default SpamDetector;