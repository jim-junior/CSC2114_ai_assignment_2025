/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/ban-ts-comment */
/* eslint-disable @typescript-eslint/no-unused-vars */
import { useState, useRef, useEffect } from "react";
import axios from "axios";
import { FiSend, FiTrash2, FiDownload, FiPlus } from "react-icons/fi";
import { AiOutlineLoading3Quarters } from "react-icons/ai";

function ChatBubble({
  role,
  text,
  image,
}: {
  role: string;
  text?: string | null;
  image?: string | null;
}) {
  const isUser = role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div
        className={`max-w-[78%] p-3 rounded-2xl shadow ${
          isUser
            ? "bg-blue-600 text-white rounded-br-none"
            : "bg-gray-100 text-gray-800 rounded-bl-none"
        }`}
      >
        {text && <div className="whitespace-pre-wrap">{text}</div>}
        {image && (
          <div className="mt-3">
            <img
              src={image}
              alt="generated"
              className="w-full rounded-lg border"
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState(() => {
    return [
      {
        id: 0,
        role: "bot",
        text: 'Hi — send a prompt and I\'ll generate an image for you. Try: "A conference hall of an international event"',
      },
    ];
  });
  const [loading, setLoading] = useState(false);
  const [steps, setSteps] = useState(10);
  const [guidance, setGuidance] = useState(2);
  const [width, setWidth] = useState(512);
  const [height, setHeight] = useState(512);
  const containerRef = useRef(null);
  const [API_BASE, setApiBase] = useState(
    "https://5fb42379abfb.ngrok-free.app"
  );

  // Auto-scroll on new message
  useEffect(() => {
    if (containerRef.current) {
      // @ts-ignore
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [messages, loading]);

  function addMessage(msg: any) {
    setMessages((m) => [...m, { ...msg, id: Date.now() }]);
  }

  async function sendPrompt() {
    if (
      API_BASE.trim() === "" ||
      !API_BASE.startsWith("http") ||
      API_BASE === null
    ) {
      alert("Please set a valid API Base URL.");
      return;
    }

    if (!prompt.trim()) return;
    const text = prompt.trim();
    addMessage({ role: "user", text });
    setPrompt("");

    setLoading(true);
    addMessage({
      role: "bot",
      text: "Generating…",
      _meta: { placeholder: true },
    });

    try {
      const payload = {
        prompt: text,
        steps,
        guidance_scale: guidance,
        width,
        height,
      };

      const res = await axios.post(`${API_BASE}/generate`, payload, {
        timeout: 120000,
      });

      // normalize response
      let imageBase64 = null;
      if (res.data?.image_base64) imageBase64 = res.data.image_base64;
      else if (res.data?.image_url) {
        // backend might return a URL
        imageBase64 = res.data.image_url;
      } else {
        // attempt fallback if backend returns raw data
        imageBase64 = res.data;
      }

      // remove the last placeholder bot message
      setMessages((prev: any) => {
        const withoutPlaceholder = prev.filter(
          (m: any) => !(m.role === "bot" && m._meta && m._meta.placeholder)
        );
        return [
          ...withoutPlaceholder,
          {
            role: "bot",
            image: imageBase64
              ? imageBase64.startsWith("data:")
                ? imageBase64
                : `data:image/png;base64,${imageBase64}`
              : null,
            text: null,
            id: Date.now(),
          },
        ];
      });
    } catch (err) {
      console.error(err);
      setMessages((prev: any) => {
        const withoutPlaceholder = prev.filter(
          (m: any) => !(m.role === "bot" && m._meta && m._meta.placeholder)
        );
        return [
          ...withoutPlaceholder,
          {
            role: "bot",
            text: `Error: ${
              // @ts-ignore
              err?.response?.data?.detail || err.message || "Request failed"
            }`,
          },
        ];
      });
    } finally {
      setLoading(false);
    }
  }

  function clearChat() {
    setMessages([
      { id: 0, role: "bot", text: "Chat cleared. Send a prompt to begin." },
    ]);
  }

  function handleKey(e: any) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendPrompt();
    }
  }

  function downloadImage(base64: string) {
    const link = document.createElement("a");
    link.href = base64;
    link.download = `sd-${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    link.remove();
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white flex items-center justify-center p-6">
      <div className="w-full max-w-3xl bg-white shadow-lg rounded-2xl overflow-hidden border">
        {/* Enhanced to allow API base URL configuration */}
        <div className="px-6 py-4">
          <label className="block text-sm font-medium mb-1">
            API Base URL:
          </label>
          <input
            type="text"
            value={API_BASE}
            onChange={(e) => setApiBase(e.target.value)}
            className="w-full p-2 rounded-md border"
            placeholder="http://localhost:8000"
          />
        </div>
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-blue-500 to-violet-500 flex items-center justify-center text-white font-bold">
              AI
            </div>
            <div>
              <div className="text-lg font-semibold">Event Gen AI</div>
              <div className="text-xs text-gray-500">
                Generate events images from prompts
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={clearChat}
              className="flex items-center gap-2 text-sm px-3 py-1 rounded-md hover:bg-gray-100"
            >
              <FiTrash2 /> Clear
            </button>
          </div>
        </div>

        {/* Main content */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Chat area */}
          <div className="md:col-span-2 flex flex-col h-[70vh]">
            <div ref={containerRef} className="flex-1 p-6 overflow-auto">
              {messages.map((msg) => (
                <ChatBubble
                  key={msg.id}
                  role={msg.role}
                  text={msg.text}
                  // @ts-ignore
                  image={msg.image}
                />
              ))}
            </div>

            <div className="px-4 py-3 border-t bg-gray-50">
              <div className="flex items-start gap-3">
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  onKeyDown={handleKey}
                  placeholder="Type a prompt... e.g. 'A keynote speaker presenting in a conference hall'"
                  className="flex-1 min-h-[48px] max-h-36 resize-none p-3 rounded-lg border focus:outline-none focus:ring-2 focus:ring-blue-300"
                />
                <div className="flex flex-col gap-2">
                  <button
                    onClick={sendPrompt}
                    disabled={loading || !prompt.trim()}
                    className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-60"
                  >
                    {loading ? (
                      <AiOutlineLoading3Quarters className="animate-spin" />
                    ) : (
                      <FiSend />
                    )}{" "}
                    Send
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Controls / settings */}
          <div className="md:col-span-1 border-l p-4 bg-slate-50">
            <div className="flex items-center justify-between mb-3">
              <div className="font-medium">Generation Settings</div>
              <div className="text-xs text-gray-500">
                Adjust for speed/quality
              </div>
            </div>

            <div className="mb-3">
              <label className="block text-sm mb-1">
                Steps: <span className="font-semibold">{steps}</span>
              </label>
              <input
                type="range"
                min="8"
                max="60"
                value={steps}
                onChange={(e) => setSteps(Number(e.target.value))}
                className="w-full"
              />
            </div>

            <div className="mb-3">
              <label className="block text-sm mb-1">
                Guidance: <span className="font-semibold">{guidance}</span>
              </label>
              <input
                type="range"
                min="1"
                max="12"
                step="0.1"
                value={guidance}
                onChange={(e) => setGuidance(Number(e.target.value))}
                className="w-full"
              />
            </div>

            <div className="mb-3 grid grid-cols-2 gap-2">
              <div>
                <label className="text-sm block mb-1">Width</label>
                <select
                  value={width}
                  onChange={(e) => setWidth(Number(e.target.value))}
                  className="w-full p-2 rounded-md border"
                >
                  <option value={256}>256</option>
                  <option value={384}>384</option>
                  <option value={512}>512</option>
                </select>
              </div>
              <div>
                <label className="text-sm block mb-1">Height</label>
                <select
                  value={height}
                  onChange={(e) => setHeight(Number(e.target.value))}
                  className="w-full p-2 rounded-md border"
                >
                  <option value={256}>256</option>
                  <option value={384}>384</option>
                  <option value={512}>512</option>
                </select>
              </div>
            </div>

            <div className="mt-4">
              <div className="text-sm font-medium mb-2">Last Image Actions</div>
              <div className="flex gap-2">
                <button
                  className="flex-1 p-2 rounded-md border flex items-center justify-center gap-2"
                  onClick={() => {
                    const last = messages
                      .slice()
                      .reverse()
                      .find((m: any) => m.image);
                    // @ts-ignore
                    if (last?.image) downloadImage(last.image);
                  }}
                >
                  <FiDownload /> Download
                </button>

                <button
                  className="p-2 rounded-md border flex items-center justify-center"
                  onClick={() => {
                    const last = messages
                      .slice()
                      .reverse()
                      .find((m: any) => m.image);
                    // @ts-ignore
                    if (last?.image) navigator.clipboard.writeText(last.image);
                  }}
                  title="Copy image data URL"
                >
                  <FiPlus />
                </button>
              </div>
            </div>

            <div className="mt-6 text-xs text-gray-500">
              This Project was created by Group SW-AI-46 for CSC2114 AI
              Assignment. Copyright © 2025. Beingana Jim Junior, Simon Peter
              Mujuni, Boonabaana Bronia. Code is open-source on{" "}
              <a
                href="https://github.com/jim-junior/CSC2114_ai_assignment_2025"
                target="_blank"
                className="text-blue-600 underline"
              >
                GitHub
              </a>
              .
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
