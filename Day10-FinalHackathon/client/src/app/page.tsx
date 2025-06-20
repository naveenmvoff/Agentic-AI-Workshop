'use client';

import { useState, useRef, useEffect } from 'react';

export default function Home() {
  const [messages, setMessages] = useState<{ text: string; type: 'user' | 'bot' }[]>([]);
  const [input, setInput] = useState('');
  const [layoutHTML, setLayoutHTML] = useState('');
  const [cssProps, setCssProps] = useState<Record<string, any>>({});
  const chatRef = useRef<HTMLDivElement>(null);

  const sendCommand = async (commandText: string) => {
    const userMessage = { text: commandText, type: 'user' } as const;
    const thinkingMessage = { text: '🧠 Thinking...', type: 'bot' } as const;

    setMessages((prev) => [...prev, userMessage, thinkingMessage]);
    setInput('');

    try {
      const response = await fetch('http://localhost:8000/process_command', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ command: commandText }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      const html = result?.data?.current_state?.layout;
      const css = result?.data?.current_state?.props;

      const botReply = result?.data?.result?.trim() ||
        (html || css ? '✅ Website updated successfully!' : "Sorry, I couldn't process your command.");

      setMessages((prev) => [
        ...prev.slice(0, -1), // Remove thinking message
        { text: botReply, type: 'bot' },
      ]);
      setLayoutHTML(html || '');
      setCssProps(css || {});
    } catch (error) {
      console.error('Error processing command:', error);
      setMessages((prev) => [
        ...prev.slice(0, -1), // Remove thinking message
        { text: '⚠️ Failed to connect to the backend.', type: 'bot' },
      ]);
    }
  };

  const handleSend = () => {
    if (!input.trim()) return;
    sendCommand(input.trim());
  };

  const handleUndo = () => {
    sendCommand('undo');
  };

  useEffect(() => {
    chatRef.current?.scrollTo(0, chatRef.current.scrollHeight);
  }, [messages]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') handleSend();
  };

  const generateInlineStyles = (props: Record<string, any>) => {
    let styles = '';
    for (const selector in props) {
      const rules = props[selector];
      if (typeof rules === 'object') {
        styles += `${selector} {\n`;
        for (const key in rules) {
          const cssKey = key.replace(/[A-Z]/g, (m) => `-${m.toLowerCase()}`);
          styles += `  ${cssKey}: ${rules[key]};\n`;
        }
        styles += `}\n`;
      }
    }
    return styles;
  };

  return (
    <main className="flex flex-col h-screen bg-gray-100 text-gray-800">
      {/* Header */}
      <div className="p-4 bg-blue-600 text-white text-xl font-bold shadow">
        ChatBot UI
      </div>

      {/* Chat Area */}
      <div
        ref={chatRef}
        className="flex-1 overflow-y-auto px-4 py-2 space-y-4"
      >
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`max-w-xs px-4 py-2 rounded-xl ${
              msg.type === 'user'
                ? 'bg-blue-500 text-white self-end ml-auto'
                : 'bg-gray-300 text-gray-900 self-start mr-auto'
            }`}
          >
            {msg.text}
          </div>
        ))}
      </div>

      {/* Input + Undo Area */}
      <div className="p-4 bg-white shadow-inner">
        <div className="flex items-center gap-2">
          <div className="flex items-center border border-gray-300 rounded-full overflow-hidden flex-1">
            <input
              type="text"
              placeholder="Type a message..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              className="flex-1 px-4 py-2 focus:outline-none"
            />
            <button
              onClick={handleSend}
              className="bg-blue-500 text-white px-5 py-2 hover:bg-blue-600 transition"
            >
              Send
            </button>
          </div>
          <button
            onClick={handleUndo}
            className="bg-yellow-500 text-white px-4 py-2 rounded-full hover:bg-yellow-600 transition"
            title="Undo last action"
          >
            Undo
          </button>
        </div>
      </div>

      {/* Preview Area */}
      {layoutHTML && (
        <div className="p-4 bg-white border-t border-gray-300 overflow-y-auto h-[50vh]">
          <h2 className="text-lg font-semibold mb-2">Live Preview:</h2>
          <style>{generateInlineStyles(cssProps)}</style>
          <div
            className="border border-gray-300 p-4 rounded bg-white"
            dangerouslySetInnerHTML={{ __html: layoutHTML }}
          />
        </div>
      )}
    </main>
  );
}



// 'use client';

// import { useState, useRef, useEffect } from 'react';

// export default function Home() {
//   const [messages, setMessages] = useState<{ text: string; type: 'user' | 'bot' }[]>([]);
//   const [input, setInput] = useState('');
//   const [layoutHTML, setLayoutHTML] = useState('');
//   const [cssProps, setCssProps] = useState<Record<string, any>>({});
//   const chatRef = useRef<HTMLDivElement>(null);

//   const handleSend = async () => {
//     if (!input.trim()) return;

//     const userMessage = { text: input, type: 'user' } as const;
//     const thinkingMessage = { text: '🧠 Thinking...', type: 'bot' } as const;

//     setMessages((prev) => [...prev, userMessage, thinkingMessage]);
//     setInput('');

//     try {
//       const response = await fetch('http://localhost:8000/process_command', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({ command: input }),
//       });

//       if (!response.ok) {
//         throw new Error(`HTTP error! status: ${response.status}`);
//       }

//       const result = await response.json();
//       const html = result?.data?.current_state?.layout;
//       const css = result?.data?.current_state?.props;

//       const botReply = result?.data?.result?.trim() ||
//         (html || css ? '✅ Website generated successfully!' : "Sorry, I couldn't process your command.");

//       setMessages((prev) => [
//         ...prev.slice(0, -1), // Remove thinking message
//         { text: botReply, type: 'bot' },
//       ]);
//       setLayoutHTML(html || '');
//       setCssProps(css || {});
//     } catch (error) {
//       console.error('Error processing command:', error);
//       setMessages((prev) => [
//         ...prev.slice(0, -1), // Remove thinking message
//         { text: '⚠️ Failed to connect to the backend.', type: 'bot' },
//       ]);
//     }
//   };

//   useEffect(() => {
//     chatRef.current?.scrollTo(0, chatRef.current.scrollHeight);
//   }, [messages]);

//   const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
//     if (e.key === 'Enter') handleSend();
//   };

//   const generateInlineStyles = (props: Record<string, any>) => {
//     let styles = '';
//     for (const selector in props) {
//       const rules = props[selector];
//       if (typeof rules === 'object') {
//         styles += `${selector} {\n`;
//         for (const key in rules) {
//           const cssKey = key.replace(/[A-Z]/g, (m) => `-${m.toLowerCase()}`);
//           styles += `  ${cssKey}: ${rules[key]};\n`;
//         }
//         styles += `}\n`;
//       }
//     }
//     return styles;
//   };

//   return (
//     <main className="flex flex-col h-screen bg-gray-100 text-gray-800">
//       {/* Header */}
//       <div className="p-4 bg-blue-600 text-white text-xl font-bold shadow">
//         ChatBot UI
//       </div>

//       {/* Chat Area */}
//       <div
//         ref={chatRef}
//         className="flex-1 overflow-y-auto px-4 py-2 space-y-4"
//       >
//         {messages.map((msg, idx) => (
//           <div
//             key={idx}
//             className={`max-w-xs px-4 py-2 rounded-xl ${
//               msg.type === 'user'
//                 ? 'bg-blue-500 text-white self-end ml-auto'
//                 : 'bg-gray-300 text-gray-900 self-start mr-auto'
//             }`}
//           >
//             {msg.text}
//           </div>
//         ))}
//       </div>

//       {/* Input Area */}
//       <div className="p-4 bg-white shadow-inner">
//         <div className="flex items-center border border-gray-300 rounded-full overflow-hidden">
//           <input
//             type="text"
//             placeholder="Type a message..."
//             value={input}
//             onChange={(e) => setInput(e.target.value)}
//             onKeyDown={handleKeyDown}
//             className="flex-1 px-4 py-2 focus:outline-none"
//           />
//           <button
//             onClick={handleSend}
//             className="bg-blue-500 text-white px-5 py-2 hover:bg-blue-600 transition"
//           >
//             Send
//           </button>
//         </div>
//       </div>

//       {/* Preview Area */}
//       {layoutHTML && (
//         <div className="p-4 bg-white border-t border-gray-300 overflow-y-auto h-[50vh]">
//           <h2 className="text-lg font-semibold mb-2">Live Preview:</h2>
//           <style>{generateInlineStyles(cssProps)}</style>
//           <div
//             className="border border-gray-300 p-4 rounded bg-white"
//             dangerouslySetInnerHTML={{ __html: layoutHTML }}
//           />
//         </div>
//       )}
//     </main>
//   );
// }
