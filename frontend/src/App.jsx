import { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, Loader2, Upload, FileText, Plus, Trash2, CheckCircle2, AlertCircle } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

function App() {
    const [input, setInput] = useState('')
    const [messages, setMessages] = useState([
        {
            role: 'assistant',
            content: 'Welcome. I am Inference Logic, your Document Intelligence Assistant. Please upload a PDF to begin the analysis.'
        }
    ])
    const [isLoading, setIsLoading] = useState(false)
    const [isUploading, setIsUploading] = useState(false)
    const [isClearing, setIsClearing] = useState(false)
    const [uploadStatus, setUploadStatus] = useState(null)
    const [showClearConfirm, setShowClearConfirm] = useState(false)
    const [loadingStep, setLoadingStep] = useState(0)

    const loadingSteps = [
        "Analyzing Query Dynamics...",
        "Executing Multi-Angle Retrieval...",
        "Synthesizing Document Context...",
        "Finalizing Intelligent Response..."
    ]

    const fileInputRef = useRef(null)
    const messagesEndRef = useRef(null)

    // Naye message aane par automatically scroll karo
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages, isLoading])

    const handleFileUpload = async (event) => {
        const file = event.target.files[0]
        if (!file) return

        setIsUploading(true)
        setUploadStatus({ type: 'info', text: `Indexing "${file.name}"...` })

        const formData = new FormData()
        formData.append('file', file)

        try {
            const response = await axios.post('http://localhost:8080/upload', formData)
            setUploadStatus({ type: 'success', text: `✓ ${file.name} indexed!` })
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `**${file.name}** has been successfully indexed. I've analyzed its structure and content. You can now ask me anything about it!`
            }])
        } catch (error) {
            setUploadStatus({ type: 'error', text: 'Upload failed. Check backend.' })
        } finally {
            setIsUploading(false)
            setTimeout(() => setUploadStatus(null), 5000)
            // Reset file input
            event.target.value = ''
        }
    }

    const handleClearIndex = async () => {
        setIsClearing(true)
        try {
            await axios.delete('http://localhost:8080/clear')
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: 'The knowledge base has been cleared. I no longer have access to previous documents.'
            }])
            setShowClearConfirm(false)
        } catch (error) {
            console.error("Failed to clear index", error)
        } finally {
            setIsClearing(false)
        }
    }

    const handleSend = async () => {
        if (!input.trim() || isLoading) return

        const question = input.trim()
        setMessages(prev => [...prev, { role: 'user', content: question }])
        setInput('')
        setIsLoading(true)
        setLoadingStep(0)

        const stepInterval = setInterval(() => {
            setLoadingStep(prev => (prev < 3 ? prev + 1 : prev))
        }, 1500)

        try {
            const response = await axios.post('http://localhost:8080/query', { question })
            clearInterval(stepInterval)
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: response.data.answer,
                sources: response.data.sources
            }])
        } catch (error) {
            clearInterval(stepInterval)
            let errorMsg = 'An unexpected error occurred while processing your request.';
            if (!error.response) {
                errorMsg = 'Connection lost. Please ensure the backend services are functional.';
            } else if (error.response.status === 500) {
                errorMsg = 'The AI engine encountered an internal error. Our logs have captured this for review.';
            }

            setMessages(prev => [...prev, {
                role: 'assistant',
                content: errorMsg
            }])
        } finally {
            setIsLoading(false)
        }
    }

    const startNewChat = () => {
        setMessages([{
            role: 'assistant',
            content: 'New chat started. Upload a PDF or ask me anything!'
        }])
        setInput('')
    }

    return (
        <div className="app-container">
            {/* Sidebar */}
            <aside className="sidebar">
                <button className="new-chat-btn" onClick={startNewChat}>
                    <Plus size={16} />
                    New Chat
                </button>

                <div className="sidebar-history">
                    <p style={{ fontSize: '11px', color: '#666', padding: '16px 12px 6px', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 'bold' }}>
                        History
                    </p>
                    <div className="history-item active">
                        <FileText size={14} style={{ marginRight: '8px' }} />
                        Current Session
                    </div>
                </div>

                <div className="sidebar-actions">
                    <div className="system-status-panel">
                        <p className="status-title">System Intelligence</p>
                        <div className="status-row">
                            <CheckCircle2 size={12} className="text-accent" />
                            <span>Multi-Query Fusion</span>
                        </div>
                        <div className="status-row">
                            <CheckCircle2 size={12} className="text-accent" />
                            <span>RRF Rank Aggregation</span>
                        </div>
                        <div className="status-row">
                            <CheckCircle2 size={12} className="text-accent" />
                            <span>Hybrid Search (Vector + Sparse)</span>
                        </div>
                    </div>

                    {!showClearConfirm ? (
                        <button className="clear-btn" onClick={() => setShowClearConfirm(true)}>
                            <Trash2 size={14} />
                            Clear Knowledge Base
                        </button>
                    ) : (
                        <div className="confirm-actions">
                            <button className="confirm-btn yes" onClick={handleClearIndex} disabled={isClearing}>
                                {isClearing ? <Loader2 size={14} className="animate-spin" /> : 'Confirm'}
                            </button>
                            <button className="confirm-btn no" onClick={() => setShowClearConfirm(false)}>Cancel</button>
                        </div>
                    )}
                </div>

                <div className="sidebar-user">
                    <div className="sidebar-avatar">
                        <User size={14} color="white" />
                    </div>
                    <div className="user-info">
                        <span className="sidebar-username">Suleiman Ahmed</span>
                        <span className="user-status">Online</span>
                    </div>
                </div>
            </aside>

            {/* Main Chat */}
            <main className="main-chat">
                {/* Header */}
                <header className="chat-header">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#10a37f' }}></div>
                        <span style={{ fontWeight: '600', fontSize: '15px' }}>Inference Logic:</span>
                    </div>
                    <AnimatePresence>
                        {uploadStatus && (
                            <motion.div
                                initial={{ opacity: 0, scale: 0.9 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.9 }}
                                className={`upload-status-badge ${uploadStatus.type}`}
                            >
                                {uploadStatus.type === 'success' ? <CheckCircle2 size={12} /> : <AlertCircle size={12} />}
                                {uploadStatus.text}
                            </motion.div>
                        )}
                    </AnimatePresence>
                </header>

                {/* Messages */}
                <div className="messages-container scrollbar-hide">
                    <AnimatePresence>
                        {messages.map((msg, idx) => (
                            <motion.div
                                key={idx}
                                initial={{ opacity: 0, y: 12 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.2 }}
                                className={`message-row ${msg.role === 'user' ? 'user-row' : 'assistant-row'}`}
                            >
                                <div className="message-wrapper">
                                    <div className={`avatar ${msg.role}`}>
                                        {msg.role === 'user' ? <User size={16} color="white" /> : <Bot size={16} color="white" />}
                                    </div>
                                    <div className="message-body">
                                        <p className="message-role-label">
                                            {msg.role === 'user' ? 'User' : 'Inference Logic'}
                                        </p>
                                        <div className="message-text">
                                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                                {msg.content}
                                            </ReactMarkdown>
                                        </div>
                                        {msg.sources && msg.sources.length > 0 && (
                                            <div className="sources-container">
                                                <p className="sources-label">Sources:</p>
                                                <div className="sources-list">
                                                    {msg.sources.map((s, i) => (
                                                        <span key={i} className="source-tag">
                                                            <FileText size={10} style={{ marginRight: '3px' }} />
                                                            {s}
                                                        </span>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </motion.div>
                        ))}
                    </AnimatePresence>

                    {/* Loading indicator */}
                    {isLoading && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="message-row assistant-row"
                        >
                            <div className="message-wrapper">
                                <div className="avatar assistant">
                                    <Loader2 className="animate-spin" size={16} color="white" />
                                </div>
                                <div className="message-body">
                                    <p className="message-role-label">System Thinking</p>
                                    <div className="thinking-status">
                                        <div className="loading-dots">
                                            <div className="dot"></div>
                                            <div className="dot"></div>
                                            <div className="dot"></div>
                                        </div>
                                        <span className="step-text">{loadingSteps[loadingStep]}</span>
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input Area */}
                <div className="input-container">
                    <div className="input-box-wrapper">
                        {/* File upload button (left side of input) */}
                        <input
                            type="file"
                            hidden
                            ref={fileInputRef}
                            onChange={handleFileUpload}
                            accept=".pdf"
                        />
                        <button
                            className="attach-btn"
                            onClick={() => fileInputRef.current.click()}
                            disabled={isUploading}
                            title="Upload PDF"
                        >
                            {isUploading
                                ? <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} />
                                : <Upload size={18} />
                            }
                        </button>

                        {/* Text input */}
                        <textarea
                            className="input-box"
                            value={input}
                            onChange={(e) => {
                                setInput(e.target.value)
                                // Auto-resize textarea
                                e.target.style.height = 'auto'
                                e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px'
                            }}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault()
                                    handleSend()
                                }
                            }}
                            placeholder="Message RAG Assistant... (Shift+Enter for new line)"
                            rows={1}
                        />

                        {/* Send button */}
                        <button
                            className="send-btn"
                            onClick={handleSend}
                            disabled={!input.trim() || isLoading}
                        >
                            <Send size={16} color="white" />
                        </button>
                    </div>
                    <p className="input-hint">
                        RAG Assistant may make mistakes. Verify important information.
                    </p>
                </div>
            </main>
        </div>
    )
}

export default App
