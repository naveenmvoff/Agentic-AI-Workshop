"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Send,
  Undo2,
  Redo2,
  Brain,
  Route,
  Wrench,
  History,
  Eye,
  Zap,
  CheckCircle,
  Clock,
  AlertCircle,
} from "lucide-react"

interface Command {
  id: string
  text: string
  timestamp: Date
  status: "processing" | "completed" | "error"
  agents: AgentStep[]
}

interface AgentStep {
  name: string
  status: "pending" | "processing" | "completed" | "error"
  description: string
  icon: any
  duration?: number
}

interface EditHistory {
  id: string
  action: string
  target: string
  timestamp: Date
  canUndo: boolean
}

export default function AgenticWebsiteEditor() {
  const [command, setCommand] = useState("")
  const [commands, setCommands] = useState<Command[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [editHistory, setEditHistory] = useState<EditHistory[]>([])
  const [activeAgent, setActiveAgent] = useState<string | null>(null)
  const [websiteContent, setWebsiteContent] = useState({
    header: { text: "Welcome to Our Website", style: "text-3xl font-bold text-gray-900" },
    button: { text: "Get Started", style: "bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700" },
    footer: { text: "© 2024 Company Name", style: "text-gray-600 text-sm" },
  })

  const inputRef = useRef<HTMLInputElement>(null)

  const agentSteps: AgentStep[] = [
    {
      name: "NL Command Parser",
      status: "pending",
      description: "Parsing natural language command",
      icon: Brain,
    },
    {
      name: "RAG Tool Router",
      status: "pending",
      description: "Routing to appropriate editing tool",
      icon: Route,
    },
    {
      name: "Change Executor",
      status: "pending",
      description: "Applying changes to website",
      icon: Wrench,
    },
    {
      name: "Memory Agent",
      status: "pending",
      description: "Logging changes for undo/redo",
      icon: History,
    },
  ]

  const simulateAgentProcessing = async (commandText: string) => {
    const newCommand: Command = {
      id: Date.now().toString(),
      text: commandText,
      timestamp: new Date(),
      status: "processing",
      agents: [...agentSteps],
    }

    setCommands((prev) => [newCommand, ...prev])
    setIsProcessing(true)

    // Simulate agent processing
    for (let i = 0; i < agentSteps.length; i++) {
      setActiveAgent(agentSteps[i].name)

      // Update agent status to processing
      setCommands((prev) =>
        prev.map((cmd) =>
          cmd.id === newCommand.id
            ? {
                ...cmd,
                agents: cmd.agents.map((agent, idx) => (idx === i ? { ...agent, status: "processing" } : agent)),
              }
            : cmd,
        ),
      )

      // Simulate processing time
      await new Promise((resolve) => setTimeout(resolve, 1000 + Math.random() * 1000))

      // Complete agent step
      setCommands((prev) =>
        prev.map((cmd) =>
          cmd.id === newCommand.id
            ? {
                ...cmd,
                agents: cmd.agents.map((agent, idx) =>
                  idx === i
                    ? { ...agent, status: "completed", duration: Math.floor(Math.random() * 500 + 200) }
                    : agent,
                ),
              }
            : cmd,
        ),
      )

      // Apply mock changes based on command
      if (i === 2) {
        // Change Executor step
        applyMockChanges(commandText)
      }
    }

    // Complete command
    setCommands((prev) => prev.map((cmd) => (cmd.id === newCommand.id ? { ...cmd, status: "completed" } : cmd)))

    setIsProcessing(false)
    setActiveAgent(null)
  }

  const applyMockChanges = (commandText: string) => {
    const lowerCommand = commandText.toLowerCase()

    if (lowerCommand.includes("header") && lowerCommand.includes("blue")) {
      setWebsiteContent((prev) => ({
        ...prev,
        header: { ...prev.header, style: "text-3xl font-bold text-blue-600" },
      }))
      addToHistory("Changed header color to blue", "header")
    }

    if (lowerCommand.includes("button") && lowerCommand.includes("green")) {
      setWebsiteContent((prev) => ({
        ...prev,
        button: { ...prev.button, style: "bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700" },
      }))
      addToHistory("Changed button color to green", "button")
    }

    if (lowerCommand.includes("footer") && lowerCommand.includes("bold")) {
      setWebsiteContent((prev) => ({
        ...prev,
        footer: { ...prev.footer, style: "text-gray-600 text-sm font-bold" },
      }))
      addToHistory("Made footer text bold", "footer")
    }
  }

  const addToHistory = (action: string, target: string) => {
    const historyItem: EditHistory = {
      id: Date.now().toString(),
      action,
      target,
      timestamp: new Date(),
      canUndo: true,
    }
    setEditHistory((prev) => [historyItem, ...prev])
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!command.trim() || isProcessing) return

    await simulateAgentProcessing(command)
    setCommand("")
  }

  const handleUndo = () => {
    if (editHistory.length > 0) {
      // Simple undo simulation - reset to default styles
      setWebsiteContent({
        header: { text: "Welcome to Our Website", style: "text-3xl font-bold text-gray-900" },
        button: { text: "Get Started", style: "bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700" },
        footer: { text: "© 2024 Company Name", style: "text-gray-600 text-sm" },
      })
      setEditHistory((prev) => prev.slice(1))
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                <Zap className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Agentic Website Editor</h1>
                <p className="text-sm text-gray-500">AI-powered conversational editing</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Button variant="outline" size="sm" onClick={handleUndo} disabled={editHistory.length === 0}>
                <Undo2 className="w-4 h-4 mr-1" />
                Undo
              </Button>
              <Button variant="outline" size="sm" disabled>
                <Redo2 className="w-4 h-4 mr-1" />
                Redo
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-120px)]">
          {/* Left Panel - Command Interface */}
          <div className="lg:col-span-1 space-y-4">
            {/* Command Input */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Brain className="w-5 h-5 text-blue-600" />
                  <span className="text-blue-600">Natural Language Commands</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSubmit} className="space-y-3">
                  <Input
                    ref={inputRef}
                    value={command}
                    onChange={(e) => setCommand(e.target.value)}
                    placeholder="e.g., Make the header blue and bold"
                    disabled={isProcessing}
                    className="w-full"
                  />
                  <Button type="submit" disabled={!command.trim() || isProcessing} className="w-full">
                    {isProcessing ? (
                      <>
                        <Clock className="w-4 h-4 mr-2 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Send className="w-4 h-4 mr-2" />
                        Execute Command
                      </>
                    )}
                  </Button>
                </form>

                {/* Quick Commands */}
                <div className="mt-4 space-y-2">
                  <p className="text-sm font-medium text-gray-700">Quick Commands:</p>
                  <div className="flex flex-wrap gap-2">
                    {["Make header blue", "Change button to green", "Make footer bold", "Add shadow to card"].map(
                      (quickCmd) => (
                        <Button
                          key={quickCmd}
                          variant="outline"
                          size="sm"
                          onClick={() => setCommand(quickCmd)}
                          disabled={isProcessing}
                          className="text-xs"
                        >
                          {quickCmd}
                        </Button>
                      ),
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Agent Workflow */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Route className="w-5 h-5 text-purple-600" />
                  <span>Agent Workflow</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {commands.length > 0 &&
                    commands[0].agents.map((agent, index) => (
                      <div key={agent.name} className="flex items-center space-x-3">
                        <div
                          className={`w-8 h-8 rounded-full flex items-center justify-center ${
                            agent.status === "completed"
                              ? "bg-green-100 text-green-600"
                              : agent.status === "processing"
                                ? "bg-blue-100 text-blue-600"
                                : agent.status === "error"
                                  ? "bg-red-100 text-red-600"
                                  : "bg-gray-100 text-gray-400"
                          }`}
                        >
                          {agent.status === "completed" ? (
                            <CheckCircle className="w-4 h-4" />
                          ) : agent.status === "processing" ? (
                            <Clock className="w-4 h-4 animate-spin" />
                          ) : agent.status === "error" ? (
                            <AlertCircle className="w-4 h-4" />
                          ) : (
                            <agent.icon className="w-4 h-4" />
                          )}
                        </div>
                        <div className="flex-1">
                          <p className="text-sm font-medium text-gray-900">{agent.name}</p>
                          <p className="text-xs text-gray-500">{agent.description}</p>
                          {agent.duration && <p className="text-xs text-green-600">{agent.duration}ms</p>}
                        </div>
                        <Badge
                          variant={
                            agent.status === "completed"
                              ? "default"
                              : agent.status === "processing"
                                ? "secondary"
                                : agent.status === "error"
                                  ? "destructive"
                                  : "outline"
                          }
                        >
                          {agent.status}
                        </Badge>
                      </div>
                    ))}
                </div>
              </CardContent>
            </Card>

            {/* Edit History */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <History className="w-5 h-5 text-orange-600" />
                  <span>Edit History</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-32">
                  {editHistory.length === 0 ? (
                    <p className="text-sm text-gray-500 text-center py-4">No edits yet</p>
                  ) : (
                    <div className="space-y-2">
                      {editHistory.map((edit) => (
                        <div key={edit.id} className="flex items-center justify-between text-sm">
                          <div>
                            <p className="font-medium text-gray-900">{edit.action}</p>
                            <p className="text-xs text-gray-500">{edit.timestamp.toLocaleTimeString()}</p>
                          </div>
                          <Badge variant="outline" className="text-xs">
                            {edit.target}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  )}
                </ScrollArea>
              </CardContent>
            </Card>
          </div>

          {/* Center Panel - Website Preview */}
          <div className="lg:col-span-2">
            <Card className="h-full">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Eye className="w-5 h-5 text-green-600" />
                  <span>Live Website Preview</span>
                  <Badge variant="secondary" className="ml-auto">
                    Real-time Updates
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="h-full">
                <div className="bg-white rounded-lg border-2 border-dashed border-gray-200 h-full p-8">
                  {/* Mock Website Content */}
                  <div className="space-y-8">
                    {/* Header */}
                    <header className="text-center">
                      <h1 className={websiteContent.header.style}>{websiteContent.header.text}</h1>
                      <p className="text-gray-600 mt-2">Experience the power of AI-driven website editing</p>
                    </header>

                    {/* Hero Section */}
                    <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-8 text-center">
                      <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                        Transform Your Website with Natural Language
                      </h2>
                      <p className="text-gray-600 mb-6">
                        Simply describe what you want to change, and our AI agents will handle the rest.
                      </p>
                      <button className={websiteContent.button.style}>{websiteContent.button.text}</button>
                    </div>

                    {/* Features Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                      {[
                        { icon: Brain, title: "Smart Parsing", desc: "AI understands your intent" },
                        { icon: Route, title: "Intelligent Routing", desc: "Routes to the right tools" },
                        { icon: Wrench, title: "Precise Execution", desc: "Makes exact changes" },
                      ].map((feature, index) => (
                        <div key={index} className="text-center p-4 rounded-lg border border-gray-200">
                          <feature.icon className="w-8 h-8 text-blue-600 mx-auto mb-3" />
                          <h3 className="font-semibold text-gray-900 mb-2">{feature.title}</h3>
                          <p className="text-sm text-gray-600">{feature.desc}</p>
                        </div>
                      ))}
                    </div>

                    {/* Footer */}
                    <footer className="text-center pt-8 border-t border-gray-200">
                      <p className={websiteContent.footer.style}>{websiteContent.footer.text}</p>
                    </footer>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* Command History Sidebar */}
      <div className="fixed right-4 top-24 w-80 max-h-96 hidden xl:block">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Command History</CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-64">
              {commands.length === 0 ? (
                <p className="text-sm text-gray-500 text-center py-4">No commands yet</p>
              ) : (
                <div className="space-y-3">
                  {commands.map((cmd) => (
                    <div key={cmd.id} className="p-3 rounded-lg border border-gray-200">
                      <div className="flex items-center justify-between mb-2">
                        <Badge
                          variant={
                            cmd.status === "completed"
                              ? "default"
                              : cmd.status === "processing"
                                ? "secondary"
                                : "destructive"
                          }
                        >
                          {cmd.status}
                        </Badge>
                        <span className="text-xs text-gray-500">{cmd.timestamp.toLocaleTimeString()}</span>
                      </div>
                      <p className="text-sm text-gray-900">{cmd.text}</p>
                    </div>
                  ))}
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}


// "use client"

// import type React from "react"

// import { useState, useRef } from "react"
// import { Button } from "@/components/ui/button"
// import { Input } from "@/components/ui/input"
// import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
// import { Badge } from "@/components/ui/badge"
// import { ScrollArea } from "@/components/ui/scroll-area"
// import {
//   Send,
//   Undo2,
//   Redo2,
//   Brain,
//   Route,
//   Wrench,
//   History,
//   Eye,
//   Zap,
//   CheckCircle,
//   Clock,
//   AlertCircle,
// } from "lucide-react"

// interface Command {
//   id: string
//   text: string
//   timestamp: Date
//   status: "processing" | "completed" | "error"
//   agents: AgentStep[]
// }

// interface AgentStep {
//   name: string
//   status: "pending" | "processing" | "completed" | "error"
//   description: string
//   icon: any
//   duration?: number
// }

// interface EditHistory {
//   id: string
//   action: string
//   target: string
//   timestamp: Date
//   canUndo: boolean
// }

// export default function AgenticWebsiteEditor() {
//   const [command, setCommand] = useState("")
//   const [commands, setCommands] = useState<Command[]>([])
//   const [isProcessing, setIsProcessing] = useState(false)
//   const [editHistory, setEditHistory] = useState<EditHistory[]>([])
//   const [activeAgent, setActiveAgent] = useState<string | null>(null)
//   const [websiteContent, setWebsiteContent] = useState({
//     header: { text: "Welcome to Our Website", style: "text-3xl font-bold text-gray-900" },
//     button: { text: "Get Started", style: "bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700" },
//     footer: { text: "© 2024 Company Name", style: "text-gray-600 text-sm" },
//   })

//   const inputRef = useRef<HTMLInputElement>(null)

//   const agentSteps: AgentStep[] = [
//     {
//       name: "NL Command Parser",
//       status: "pending",
//       description: "Parsing natural language command",
//       icon: Brain,
//     },
//     {
//       name: "RAG Tool Router",
//       status: "pending",
//       description: "Routing to appropriate editing tool",
//       icon: Route,
//     },
//     {
//       name: "Change Executor",
//       status: "pending",
//       description: "Applying changes to website",
//       icon: Wrench,
//     },
//     {
//       name: "Memory Agent",
//       status: "pending",
//       description: "Logging changes for undo/redo",
//       icon: History,
//     },
//   ]

//   const simulateAgentProcessing = async (commandText: string) => {
//     const newCommand: Command = {
//       id: Date.now().toString(),
//       text: commandText,
//       timestamp: new Date(),
//       status: "processing",
//       agents: [...agentSteps],
//     }

//     setCommands((prev) => [newCommand, ...prev])
//     setIsProcessing(true)

//     // Simulate agent processing
//     for (let i = 0; i < agentSteps.length; i++) {
//       setActiveAgent(agentSteps[i].name)

//       // Update agent status to processing
//       setCommands((prev) =>
//         prev.map((cmd) =>
//           cmd.id === newCommand.id
//             ? {
//                 ...cmd,
//                 agents: cmd.agents.map((agent, idx) => (idx === i ? { ...agent, status: "processing" } : agent)),
//               }
//             : cmd,
//         ),
//       )

//       // Simulate processing time
//       await new Promise((resolve) => setTimeout(resolve, 1000 + Math.random() * 1000))

//       // Complete agent step
//       setCommands((prev) =>
//         prev.map((cmd) =>
//           cmd.id === newCommand.id
//             ? {
//                 ...cmd,
//                 agents: cmd.agents.map((agent, idx) =>
//                   idx === i
//                     ? { ...agent, status: "completed", duration: Math.floor(Math.random() * 500 + 200) }
//                     : agent,
//                 ),
//               }
//             : cmd,
//         ),
//       )

//       // Apply mock changes based on command
//       if (i === 2) {
//         // Change Executor step
//         applyMockChanges(commandText)
//       }
//     }

//     // Complete command
//     setCommands((prev) => prev.map((cmd) => (cmd.id === newCommand.id ? { ...cmd, status: "completed" } : cmd)))

//     setIsProcessing(false)
//     setActiveAgent(null)
//   }

//   const applyMockChanges = (commandText: string) => {
//     const lowerCommand = commandText.toLowerCase()

//     if (lowerCommand.includes("header") && lowerCommand.includes("blue")) {
//       setWebsiteContent((prev) => ({
//         ...prev,
//         header: { ...prev.header, style: "text-3xl font-bold text-blue-600" },
//       }))
//       addToHistory("Changed header color to blue", "header")
//     }

//     if (lowerCommand.includes("button") && lowerCommand.includes("green")) {
//       setWebsiteContent((prev) => ({
//         ...prev,
//         button: { ...prev.button, style: "bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700" },
//       }))
//       addToHistory("Changed button color to green", "button")
//     }

//     if (lowerCommand.includes("footer") && lowerCommand.includes("bold")) {
//       setWebsiteContent((prev) => ({
//         ...prev,
//         footer: { ...prev.footer, style: "text-gray-600 text-sm font-bold" },
//       }))
//       addToHistory("Made footer text bold", "footer")
//     }
//   }

//   const addToHistory = (action: string, target: string) => {
//     const historyItem: EditHistory = {
//       id: Date.now().toString(),
//       action,
//       target,
//       timestamp: new Date(),
//       canUndo: true,
//     }
//     setEditHistory((prev) => [historyItem, ...prev])
//   }

//   const handleSubmit = async (e: React.FormEvent) => {
//     e.preventDefault()
//     if (!command.trim() || isProcessing) return

//     await simulateAgentProcessing(command)
//     setCommand("")
//   }

//   const handleUndo = () => {
//     if (editHistory.length > 0) {
//       // Simple undo simulation - reset to default styles
//       setWebsiteContent({
//         header: { text: "Welcome to Our Website", style: "text-3xl font-bold text-gray-900" },
//         button: { text: "Get Started", style: "bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700" },
//         footer: { text: "© 2024 Company Name", style: "text-gray-600 text-sm" },
//       })
//       setEditHistory((prev) => prev.slice(1))
//     }
//   }

//   return (
//     <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
//       {/* Header */}
//       <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
//         <div className="container mx-auto px-4 py-4">
//           <div className="flex items-center justify-between">
//             <div className="flex items-center space-x-3">
//               <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
//                 <Zap className="w-5 h-5 text-white" />
//               </div>
//               <div>
//                 <h1 className="text-xl font-bold text-gray-900">Agentic Website Editor</h1>
//                 <p className="text-sm text-gray-500">AI-powered conversational editing</p>
//               </div>
//             </div>
//             <div className="flex items-center space-x-2">
//               <Button variant="outline" size="sm" onClick={handleUndo} disabled={editHistory.length === 0}>
//                 <Undo2 className="w-4 h-4 mr-1" />
//                 Undo
//               </Button>
//               <Button variant="outline" size="sm" disabled>
//                 <Redo2 className="w-4 h-4 mr-1" />
//                 Redo
//               </Button>
//             </div>
//           </div>
//         </div>
//       </header>

//       <div className="container mx-auto px-4 py-6">
//         <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-120px)]">
//           {/* Left Panel - Command Interface */}
//           <div className="lg:col-span-1 space-y-4">
//             {/* Command Input */}
//             <Card>
//               <CardHeader>
//                 <CardTitle className="flex items-center space-x-2">
//                   <Brain className="w-5 h-5 text-blue-600" />
//                   <span>Natural Language Commands</span>
//                 </CardTitle>
//               </CardHeader>
//               <CardContent>
//                 <form onSubmit={handleSubmit} className="space-y-3">
//                   <Input
//                     ref={inputRef}
//                     value={command}
//                     onChange={(e) => setCommand(e.target.value)}
//                     placeholder="e.g., Make the header blue and bold"
//                     disabled={isProcessing}
//                     className="w-full"
//                   />
//                   <Button type="submit" disabled={!command.trim() || isProcessing} className="w-full">
//                     {isProcessing ? (
//                       <>
//                         <Clock className="w-4 h-4 mr-2 animate-spin" />
//                         Processing...
//                       </>
//                     ) : (
//                       <>
//                         <Send className="w-4 h-4 mr-2" />
//                         Execute Command
//                       </>
//                     )}
//                   </Button>
//                 </form>

//                 {/* Quick Commands */}
//                 <div className="mt-4 space-y-2">
//                   <p className="text-sm font-medium text-gray-700">Quick Commands:</p>
//                   <div className="flex flex-wrap gap-2">
//                     {["Make header blue", "Change button to green", "Make footer bold", "Add shadow to card"].map(
//                       (quickCmd) => (
//                         <Button
//                           key={quickCmd}
//                           variant="outline"
//                           size="sm"
//                           onClick={() => setCommand(quickCmd)}
//                           disabled={isProcessing}
//                           className="text-xs"
//                         >
//                           {quickCmd}
//                         </Button>
//                       ),
//                     )}
//                   </div>
//                 </div>
//               </CardContent>
//             </Card>

//             {/* Agent Workflow */}
//             <Card>
//               <CardHeader>
//                 <CardTitle className="flex items-center space-x-2">
//                   <Route className="w-5 h-5 text-purple-600" />
//                   <span>Agent Workflow</span>
//                 </CardTitle>
//               </CardHeader>
//               <CardContent>
//                 <div className="space-y-3">
//                   {commands.length > 0 &&
//                     commands[0].agents.map((agent, index) => (
//                       <div key={agent.name} className="flex items-center space-x-3">
//                         <div
//                           className={`w-8 h-8 rounded-full flex items-center justify-center ${
//                             agent.status === "completed"
//                               ? "bg-green-100 text-green-600"
//                               : agent.status === "processing"
//                                 ? "bg-blue-100 text-blue-600"
//                                 : agent.status === "error"
//                                   ? "bg-red-100 text-red-600"
//                                   : "bg-gray-100 text-gray-400"
//                           }`}
//                         >
//                           {agent.status === "completed" ? (
//                             <CheckCircle className="w-4 h-4" />
//                           ) : agent.status === "processing" ? (
//                             <Clock className="w-4 h-4 animate-spin" />
//                           ) : agent.status === "error" ? (
//                             <AlertCircle className="w-4 h-4" />
//                           ) : (
//                             <agent.icon className="w-4 h-4" />
//                           )}
//                         </div>
//                         <div className="flex-1">
//                           <p className="text-sm font-medium text-gray-900">{agent.name}</p>
//                           <p className="text-xs text-gray-500">{agent.description}</p>
//                           {agent.duration && <p className="text-xs text-green-600">{agent.duration}ms</p>}
//                         </div>
//                         <Badge
//                           variant={
//                             agent.status === "completed"
//                               ? "default"
//                               : agent.status === "processing"
//                                 ? "secondary"
//                                 : agent.status === "error"
//                                   ? "destructive"
//                                   : "outline"
//                           }
//                         >
//                           {agent.status}
//                         </Badge>
//                       </div>
//                     ))}
//                 </div>
//               </CardContent>
//             </Card>

//             {/* Edit History */}
//             <Card>
//               <CardHeader>
//                 <CardTitle className="flex items-center space-x-2">
//                   <History className="w-5 h-5 text-orange-600" />
//                   <span>Edit History</span>
//                 </CardTitle>
//               </CardHeader>
//               <CardContent>
//                 <ScrollArea className="h-32">
//                   {editHistory.length === 0 ? (
//                     <p className="text-sm text-gray-500 text-center py-4">No edits yet</p>
//                   ) : (
//                     <div className="space-y-2">
//                       {editHistory.map((edit) => (
//                         <div key={edit.id} className="flex items-center justify-between text-sm">
//                           <div>
//                             <p className="font-medium text-gray-900">{edit.action}</p>
//                             <p className="text-xs text-gray-500">{edit.timestamp.toLocaleTimeString()}</p>
//                           </div>
//                           <Badge variant="outline" className="text-xs">
//                             {edit.target}
//                           </Badge>
//                         </div>
//                       ))}
//                     </div>
//                   )}
//                 </ScrollArea>
//               </CardContent>
//             </Card>
//           </div>

//           {/* Center Panel - Website Preview */}
//           <div className="lg:col-span-2">
//             <Card className="h-full">
//               <CardHeader>
//                 <CardTitle className="flex items-center space-x-2">
//                   <Eye className="w-5 h-5 text-green-600" />
//                   <span>Live Website Preview</span>
//                   <Badge variant="secondary" className="ml-auto">
//                     Real-time Updates
//                   </Badge>
//                 </CardTitle>
//               </CardHeader>
//               <CardContent className="h-full">
//                 <div className="bg-white rounded-lg border-2 border-dashed border-gray-200 h-full p-8">
//                   {/* Mock Website Content */}
//                   <div className="space-y-8">
//                     {/* Header */}
//                     <header className="text-center">
//                       <h1 className={websiteContent.header.style}>{websiteContent.header.text}</h1>
//                       <p className="text-gray-600 mt-2">Experience the power of AI-driven website editing</p>
//                     </header>

//                     {/* Hero Section */}
//                     <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-8 text-center">
//                       <h2 className="text-2xl font-semibold text-gray-900 mb-4">
//                         Transform Your Website with Natural Language
//                       </h2>
//                       <p className="text-gray-600 mb-6">
//                         Simply describe what you want to change, and our AI agents will handle the rest.
//                       </p>
//                       <button className={websiteContent.button.style}>{websiteContent.button.text}</button>
//                     </div>

//                     {/* Features Grid */}
//                     <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
//                       {[
//                         { icon: Brain, title: "Smart Parsing", desc: "AI understands your intent" },
//                         { icon: Route, title: "Intelligent Routing", desc: "Routes to the right tools" },
//                         { icon: Wrench, title: "Precise Execution", desc: "Makes exact changes" },
//                       ].map((feature, index) => (
//                         <div key={index} className="text-center p-4 rounded-lg border border-gray-200">
//                           <feature.icon className="w-8 h-8 text-blue-600 mx-auto mb-3" />
//                           <h3 className="font-semibold text-gray-900 mb-2">{feature.title}</h3>
//                           <p className="text-sm text-gray-600">{feature.desc}</p>
//                         </div>
//                       ))}
//                     </div>

//                     {/* Footer */}
//                     <footer className="text-center pt-8 border-t border-gray-200">
//                       <p className={websiteContent.footer.style}>{websiteContent.footer.text}</p>
//                     </footer>
//                   </div>
//                 </div>
//               </CardContent>
//             </Card>
//           </div>
//         </div>
//       </div>

//       {/* Command History Sidebar */}
//       <div className="fixed right-4 top-24 w-80 max-h-96 hidden xl:block">
//         <Card>
//           <CardHeader>
//             <CardTitle className="text-sm">Command History</CardTitle>
//           </CardHeader>
//           <CardContent>
//             <ScrollArea className="h-64">
//               {commands.length === 0 ? (
//                 <p className="text-sm text-gray-500 text-center py-4">No commands yet</p>
//               ) : (
//                 <div className="space-y-3">
//                   {commands.map((cmd) => (
//                     <div key={cmd.id} className="p-3 rounded-lg border border-gray-200">
//                       <div className="flex items-center justify-between mb-2">
//                         <Badge
//                           variant={
//                             cmd.status === "completed"
//                               ? "default"
//                               : cmd.status === "processing"
//                                 ? "secondary"
//                                 : "destructive"
//                           }
//                         >
//                           {cmd.status}
//                         </Badge>
//                         <span className="text-xs text-gray-500">{cmd.timestamp.toLocaleTimeString()}</span>
//                       </div>
//                       <p className="text-sm text-gray-900">{cmd.text}</p>
//                     </div>
//                   ))}
//                 </div>
//               )}
//             </ScrollArea>
//           </CardContent>
//         </Card>
//       </div>
//     </div>
//   )
// }








// // "use client"

// // import type React from "react"

// // import { useState, useRef } from "react"
// // import { Button } from "@/components/ui/button"
// // import { Input } from "@/components/ui/input"
// // import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
// // import { Badge } from "@/components/ui/badge"
// // import { ScrollArea } from "@/components/ui/scroll-area"
// // import {
// //   Send,
// //   Undo2,
// //   Redo2,
// //   Brain,
// //   Route,
// //   Wrench,
// //   History,
// //   Eye,
// //   Zap,
// //   CheckCircle,
// //   Clock,
// //   AlertCircle,
// // } from "lucide-react"

// // interface Command {
// //   id: string
// //   text: string
// //   timestamp: Date
// //   status: "processing" | "completed" | "error"
// //   agents: AgentStep[]
// // }

// // interface AgentStep {
// //   name: string
// //   status: "pending" | "processing" | "completed" | "error"
// //   description: string
// //   icon: any
// //   duration?: number
// // }

// // interface EditHistory {
// //   id: string
// //   action: string
// //   target: string
// //   timestamp: Date
// //   canUndo: boolean
// // }

// // export default function AgenticWebsiteEditor() {
// //   const [command, setCommand] = useState("")
// //   const [commands, setCommands] = useState<Command[]>([])
// //   const [isProcessing, setIsProcessing] = useState(false)
// //   const [editHistory, setEditHistory] = useState<EditHistory[]>([])
// //   const [activeAgent, setActiveAgent] = useState<string | null>(null)
// //   const [websiteContent, setWebsiteContent] = useState({
// //     header: { text: "Welcome to Our Website", style: "text-3xl font-bold text-gray-900" },
// //     button: { text: "Get Started", style: "bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700" },
// //     footer: { text: "© 2024 Company Name", style: "text-gray-600 text-sm" },
// //   })

// //   const inputRef = useRef<HTMLInputElement>(null)

// //   const agentSteps: AgentStep[] = [
// //     {
// //       name: "NL Command Parser",
// //       status: "pending",
// //       description: "Parsing natural language command",
// //       icon: Brain,
// //     },
// //     {
// //       name: "RAG Tool Router",
// //       status: "pending",
// //       description: "Routing to appropriate editing tool",
// //       icon: Route,
// //     },
// //     {
// //       name: "Change Executor",
// //       status: "pending",
// //       description: "Applying changes to website",
// //       icon: Wrench,
// //     },
// //     {
// //       name: "Memory Agent",
// //       status: "pending",
// //       description: "Logging changes for undo/redo",
// //       icon: History,
// //     },
// //   ]

// //   const simulateAgentProcessing = async (commandText: string) => {
// //     const newCommand: Command = {
// //       id: Date.now().toString(),
// //       text: commandText,
// //       timestamp: new Date(),
// //       status: "processing",
// //       agents: [...agentSteps],
// //     }

// //     setCommands((prev) => [newCommand, ...prev])
// //     setIsProcessing(true)

// //     // Simulate agent processing
// //     for (let i = 0; i < agentSteps.length; i++) {
// //       setActiveAgent(agentSteps[i].name)

// //       // Update agent status to processing
// //       setCommands((prev) =>
// //         prev.map((cmd) =>
// //           cmd.id === newCommand.id
// //             ? {
// //                 ...cmd,
// //                 agents: cmd.agents.map((agent, idx) => (idx === i ? { ...agent, status: "processing" } : agent)),
// //               }
// //             : cmd,
// //         ),
// //       )

// //       // Simulate processing time
// //       await new Promise((resolve) => setTimeout(resolve, 1000 + Math.random() * 1000))

// //       // Complete agent step
// //       setCommands((prev) =>
// //         prev.map((cmd) =>
// //           cmd.id === newCommand.id
// //             ? {
// //                 ...cmd,
// //                 agents: cmd.agents.map((agent, idx) =>
// //                   idx === i
// //                     ? { ...agent, status: "completed", duration: Math.floor(Math.random() * 500 + 200) }
// //                     : agent,
// //                 ),
// //               }
// //             : cmd,
// //         ),
// //       )

// //       // Apply mock changes based on command
// //       if (i === 2) {
// //         // Change Executor step
// //         applyMockChanges(commandText)
// //       }
// //     }

// //     // Complete command
// //     setCommands((prev) => prev.map((cmd) => (cmd.id === newCommand.id ? { ...cmd, status: "completed" } : cmd)))

// //     setIsProcessing(false)
// //     setActiveAgent(null)
// //   }

// //   const applyMockChanges = (commandText: string) => {
// //     const lowerCommand = commandText.toLowerCase()

// //     if (lowerCommand.includes("header") && lowerCommand.includes("blue")) {
// //       setWebsiteContent((prev) => ({
// //         ...prev,
// //         header: { ...prev.header, style: "text-3xl font-bold text-blue-600" },
// //       }))
// //       addToHistory("Changed header color to blue", "header")
// //     }

// //     if (lowerCommand.includes("button") && lowerCommand.includes("green")) {
// //       setWebsiteContent((prev) => ({
// //         ...prev,
// //         button: { ...prev.button, style: "bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700" },
// //       }))
// //       addToHistory("Changed button color to green", "button")
// //     }

// //     if (lowerCommand.includes("footer") && lowerCommand.includes("bold")) {
// //       setWebsiteContent((prev) => ({
// //         ...prev,
// //         footer: { ...prev.footer, style: "text-gray-600 text-sm font-bold" },
// //       }))
// //       addToHistory("Made footer text bold", "footer")
// //     }
// //   }

// //   const addToHistory = (action: string, target: string) => {
// //     const historyItem: EditHistory = {
// //       id: Date.now().toString(),
// //       action,
// //       target,
// //       timestamp: new Date(),
// //       canUndo: true,
// //     }
// //     setEditHistory((prev) => [historyItem, ...prev])
// //   }

// //   const handleSubmit = async (e: React.FormEvent) => {
// //     e.preventDefault()
// //     if (!command.trim() || isProcessing) return

// //     await simulateAgentProcessing(command)
// //     setCommand("")
// //   }

// //   const handleUndo = () => {
// //     if (editHistory.length > 0) {
// //       // Simple undo simulation - reset to default styles
// //       setWebsiteContent({
// //         header: { text: "Welcome to Our Website", style: "text-3xl font-bold text-gray-900" },
// //         button: { text: "Get Started", style: "bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700" },
// //         footer: { text: "© 2024 Company Name", style: "text-gray-600 text-sm" },
// //       })
// //       setEditHistory((prev) => prev.slice(1))
// //     }
// //   }

// //   return (
// //     <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
// //       {/* Header */}
// //       <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
// //         <div className="container mx-auto px-4 py-4">
// //           <div className="flex items-center justify-between">
// //             <div className="flex items-center space-x-3">
// //               <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
// //                 <Zap className="w-5 h-5 text-white" />
// //               </div>
// //               <div>
// //                 <h1 className="text-xl font-bold text-gray-900">Agentic Website Editor</h1>
// //                 <p className="text-sm text-gray-500">AI-powered conversational editing</p>
// //               </div>
// //             </div>
// //             <div className="flex items-center space-x-2">
// //               <Button variant="outline" size="sm" onClick={handleUndo} disabled={editHistory.length === 0}>
// //                 <Undo2 className="w-4 h-4 mr-1" />
// //                 Undo
// //               </Button>
// //               <Button variant="outline" size="sm" disabled>
// //                 <Redo2 className="w-4 h-4 mr-1" />
// //                 Redo
// //               </Button>
// //             </div>
// //           </div>
// //         </div>
// //       </header>

// //       <div className="container mx-auto px-4 py-6">
// //         <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-120px)]">
// //           {/* Left Panel - Command Interface */}
// //           <div className="lg:col-span-1 space-y-4">
// //             {/* Command Input */}
// //             <Card>
// //               <CardHeader>
// //                 <CardTitle className="flex items-center space-x-2">
// //                   <Brain className="w-5 h-5 text-blue-600" />
// //                   <span>Natural Language Commands</span>
// //                 </CardTitle>
// //               </CardHeader>
// //               <CardContent>
// //                 <form onSubmit={handleSubmit} className="space-y-3">
// //                   <Input
// //                     ref={inputRef}
// //                     value={command}
// //                     onChange={(e) => setCommand(e.target.value)}
// //                     placeholder="e.g., Make the header blue and bold"
// //                     disabled={isProcessing}
// //                     className="w-full"
// //                   />
// //                   <Button type="submit" disabled={!command.trim() || isProcessing} className="w-full">
// //                     {isProcessing ? (
// //                       <>
// //                         <Clock className="w-4 h-4 mr-2 animate-spin" />
// //                         Processing...
// //                       </>
// //                     ) : (
// //                       <>
// //                         <Send className="w-4 h-4 mr-2" />
// //                         Execute Command
// //                       </>
// //                     )}
// //                   </Button>
// //                 </form>

// //                 {/* Quick Commands */}
// //                 <div className="mt-4 space-y-2">
// //                   <p className="text-sm font-medium text-gray-700">Quick Commands:</p>
// //                   <div className="flex flex-wrap gap-2">
// //                     {["Make header blue", "Change button to green", "Make footer bold", "Add shadow to card"].map(
// //                       (quickCmd) => (
// //                         <Button
// //                           key={quickCmd}
// //                           variant="outline"
// //                           size="sm"
// //                           onClick={() => setCommand(quickCmd)}
// //                           disabled={isProcessing}
// //                           className="text-xs"
// //                         >
// //                           {quickCmd}
// //                         </Button>
// //                       ),
// //                     )}
// //                   </div>
// //                 </div>
// //               </CardContent>
// //             </Card>

// //             {/* Agent Workflow */}
// //             <Card>
// //               <CardHeader>
// //                 <CardTitle className="flex items-center space-x-2">
// //                   <Route className="w-5 h-5 text-purple-600" />
// //                   <span>Agent Workflow</span>
// //                 </CardTitle>
// //               </CardHeader>
// //               <CardContent>
// //                 <div className="space-y-3">
// //                   {commands.length > 0 &&
// //                     commands[0].agents.map((agent, index) => (
// //                       <div key={agent.name} className="flex items-center space-x-3">
// //                         <div
// //                           className={`w-8 h-8 rounded-full flex items-center justify-center ${
// //                             agent.status === "completed"
// //                               ? "bg-green-100 text-green-600"
// //                               : agent.status === "processing"
// //                                 ? "bg-blue-100 text-blue-600"
// //                                 : agent.status === "error"
// //                                   ? "bg-red-100 text-red-600"
// //                                   : "bg-gray-100 text-gray-400"
// //                           }`}
// //                         >
// //                           {agent.status === "completed" ? (
// //                             <CheckCircle className="w-4 h-4" />
// //                           ) : agent.status === "processing" ? (
// //                             <Clock className="w-4 h-4 animate-spin" />
// //                           ) : agent.status === "error" ? (
// //                             <AlertCircle className="w-4 h-4" />
// //                           ) : (
// //                             <agent.icon className="w-4 h-4" />
// //                           )}
// //                         </div>
// //                         <div className="flex-1">
// //                           <p className="text-sm font-medium text-gray-900">{agent.name}</p>
// //                           <p className="text-xs text-gray-500">{agent.description}</p>
// //                           {agent.duration && <p className="text-xs text-green-600">{agent.duration}ms</p>}
// //                         </div>
// //                         <Badge
// //                           variant={
// //                             agent.status === "completed"
// //                               ? "default"
// //                               : agent.status === "processing"
// //                                 ? "secondary"
// //                                 : agent.status === "error"
// //                                   ? "destructive"
// //                                   : "outline"
// //                           }
// //                         >
// //                           {agent.status}
// //                         </Badge>
// //                       </div>
// //                     ))}
// //                 </div>
// //               </CardContent>
// //             </Card>

// //             {/* Edit History */}
// //             <Card>
// //               <CardHeader>
// //                 <CardTitle className="flex items-center space-x-2">
// //                   <History className="w-5 h-5 text-orange-600" />
// //                   <span>Edit History</span>
// //                 </CardTitle>
// //               </CardHeader>
// //               <CardContent>
// //                 <ScrollArea className="h-32">
// //                   {editHistory.length === 0 ? (
// //                     <p className="text-sm text-gray-500 text-center py-4">No edits yet</p>
// //                   ) : (
// //                     <div className="space-y-2">
// //                       {editHistory.map((edit) => (
// //                         <div key={edit.id} className="flex items-center justify-between text-sm">
// //                           <div>
// //                             <p className="font-medium text-gray-900">{edit.action}</p>
// //                             <p className="text-xs text-gray-500">{edit.timestamp.toLocaleTimeString()}</p>
// //                           </div>
// //                           <Badge variant="outline" className="text-xs">
// //                             {edit.target}
// //                           </Badge>
// //                         </div>
// //                       ))}
// //                     </div>
// //                   )}
// //                 </ScrollArea>
// //               </CardContent>
// //             </Card>
// //           </div>

// //           {/* Center Panel - Website Preview */}
// //           <div className="lg:col-span-2">
// //             <Card className="h-full">
// //               <CardHeader>
// //                 <CardTitle className="flex items-center space-x-2">
// //                   <Eye className="w-5 h-5 text-green-600" />
// //                   <span>Live Website Preview</span>
// //                   <Badge variant="secondary" className="ml-auto">
// //                     Real-time Updates
// //                   </Badge>
// //                 </CardTitle>
// //               </CardHeader>
// //               <CardContent className="h-full">
// //                 <div className="bg-white rounded-lg border-2 border-dashed border-gray-200 h-full p-8">
// //                   {/* Mock Website Content */}
// //                   <div className="space-y-8">
// //                     {/* Header */}
// //                     <header className="text-center">
// //                       <h1 className={websiteContent.header.style}>{websiteContent.header.text}</h1>
// //                       <p className="text-gray-600 mt-2">Experience the power of AI-driven website editing</p>
// //                     </header>

// //                     {/* Hero Section */}
// //                     <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-8 text-center">
// //                       <h2 className="text-2xl font-semibold text-gray-900 mb-4">
// //                         Transform Your Website with Natural Language
// //                       </h2>
// //                       <p className="text-gray-600 mb-6">
// //                         Simply describe what you want to change, and our AI agents will handle the rest.
// //                       </p>
// //                       <button className={websiteContent.button.style}>{websiteContent.button.text}</button>
// //                     </div>

// //                     {/* Features Grid */}
// //                     <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
// //                       {[
// //                         { icon: Brain, title: "Smart Parsing", desc: "AI understands your intent" },
// //                         { icon: Route, title: "Intelligent Routing", desc: "Routes to the right tools" },
// //                         { icon: Wrench, title: "Precise Execution", desc: "Makes exact changes" },
// //                       ].map((feature, index) => (
// //                         <div key={index} className="text-center p-4 rounded-lg border border-gray-200">
// //                           <feature.icon className="w-8 h-8 text-blue-600 mx-auto mb-3" />
// //                           <h3 className="font-semibold text-gray-900 mb-2">{feature.title}</h3>
// //                           <p className="text-sm text-gray-600">{feature.desc}</p>
// //                         </div>
// //                       ))}
// //                     </div>

// //                     {/* Footer */}
// //                     <footer className="text-center pt-8 border-t border-gray-200">
// //                       <p className={websiteContent.footer.style}>{websiteContent.footer.text}</p>
// //                     </footer>
// //                   </div>
// //                 </div>
// //               </CardContent>
// //             </Card>
// //           </div>
// //         </div>
// //       </div>

// //       {/* Command History Sidebar */}
// //       <div className="fixed right-4 top-24 w-80 max-h-96 hidden xl:block">
// //         <Card>
// //           <CardHeader>
// //             <CardTitle className="text-sm">Command History</CardTitle>
// //           </CardHeader>
// //           <CardContent>
// //             <ScrollArea className="h-64">
// //               {commands.length === 0 ? (
// //                 <p className="text-sm text-gray-500 text-center py-4">No commands yet</p>
// //               ) : (
// //                 <div className="space-y-3">
// //                   {commands.map((cmd) => (
// //                     <div key={cmd.id} className="p-3 rounded-lg border border-gray-200">
// //                       <div className="flex items-center justify-between mb-2">
// //                         <Badge
// //                           variant={
// //                             cmd.status === "completed"
// //                               ? "default"
// //                               : cmd.status === "processing"
// //                                 ? "secondary"
// //                                 : "destructive"
// //                           }
// //                         >
// //                           {cmd.status}
// //                         </Badge>
// //                         <span className="text-xs text-gray-500">{cmd.timestamp.toLocaleTimeString()}</span>
// //                       </div>
// //                       <p className="text-sm text-gray-900">{cmd.text}</p>
// //                     </div>
// //                   ))}
// //                 </div>
// //               )}
// //             </ScrollArea>
// //           </CardContent>
// //         </Card>
// //       </div>
// //     </div>
// //   )
// // }




// // // import Image from "next/image";

// // // export default function Home() {
// // //   return (
// // //     <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
// // //       <main className="flex flex-col gap-[32px] row-start-2 items-center sm:items-start">
// // //         <Image
// // //           className="dark:invert"
// // //           src="/next.svg"
// // //           alt="Next.js logo"
// // //           width={180}
// // //           height={38}
// // //           priority
// // //         />
// // //         <ol className="list-inside list-decimal text-sm/6 text-center sm:text-left font-[family-name:var(--font-geist-mono)]">
// // //           <li className="mb-2 tracking-[-.01em]">
// // //             Get started by editing{" "}
// // //             <code className="bg-black/[.05] dark:bg-white/[.06] px-1 py-0.5 rounded font-[family-name:var(--font-geist-mono)] font-semibold">
// // //               src/app/page.tsx
// // //             </code>
// // //             .
// // //           </li>
// // //           <li className="tracking-[-.01em]">
// // //             Save and see your changes instantly.
// // //           </li>
// // //         </ol>

// // //         <div className="flex gap-4 items-center flex-col sm:flex-row">
// // //           <a
// // //             className="rounded-full border border-solid border-transparent transition-colors flex items-center justify-center bg-gray text-background gap-2 hover:bg-[#383838] dark:hover:bg-[#ccc] font-medium text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5 sm:w-auto"
// // //             href="https://vercel.com/new?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
// // //             target="_blank"
// // //             rel="noopener noreferrer"
// // //           >
// // //             <Image
// // //               className="dark:invert"
// // //               src="/vercel.svg"
// // //               alt="Vercel logomark"
// // //               width={20}
// // //               height={20}
// // //             />
// // //             Deploy now
// // //           </a>
// // //           <a
// // //             className="rounded-full border border-solid border-black/[.08] dark:border-white/[.145] transition-colors flex items-center justify-center hover:bg-[#f2f2f2] dark:hover:bg-[#1a1a1a] hover:border-transparent font-medium text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5 w-full sm:w-auto md:w-[158px]"
// // //             href="https://nextjs.org/docs?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
// // //             target="_blank"
// // //             rel="noopener noreferrer"
// // //           >
// // //             Read our docs
// // //           </a>
// // //         </div>
// // //       </main>
// // //       <footer className="row-start-3 flex gap-[24px] flex-wrap items-center justify-center">
// // //         <a
// // //           className="flex items-center gap-2 hover:underline hover:underline-offset-4"
// // //           href="https://nextjs.org/learn?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
// // //           target="_blank"
// // //           rel="noopener noreferrer"
// // //         >
// // //           <Image
// // //             aria-hidden
// // //             src="/file.svg"
// // //             alt="File icon"
// // //             width={16}
// // //             height={16}
// // //           />
// // //           Learn
// // //         </a>
// // //         <a
// // //           className="flex items-center gap-2 hover:underline hover:underline-offset-4"
// // //           href="https://vercel.com/templates?framework=next.js&utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
// // //           target="_blank"
// // //           rel="noopener noreferrer"
// // //         >
// // //           <Image
// // //             aria-hidden
// // //             src="/window.svg"
// // //             alt="Window icon"
// // //             width={16}
// // //             height={16}
// // //           />
// // //           Examples
// // //         </a>
// // //         <a
// // //           className="flex items-center gap-2 hover:underline hover:underline-offset-4"
// // //           href="https://nextjs.org?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
// // //           target="_blank"
// // //           rel="noopener noreferrer"
// // //         >
// // //           <Image
// // //             aria-hidden
// // //             src="/globe.svg"
// // //             alt="Globe icon"
// // //             width={16}
// // //             height={16}
// // //           />
// // //           Go to nextjs.org →
// // //         </a>
// // //       </footer>
// // //     </div>
// // //   );
// // // }
