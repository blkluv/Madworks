import { promises as fs } from 'fs'
import path from 'path'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  attachments?: Array<{ type: string; url: string; variant?: string; label?: string; size?: string }>
}

export interface Conversation {
  id: string
  title: string
  messages: Message[]
  createdAt: string
  updatedAt: string
  analysis?: any
}

const BASE_DIR = path.join(process.cwd(), '.data', 'conversations')

function emailToFile(email: string) {
  const safe = email.replace(/[^a-zA-Z0-9_-]+/g, '_')
  return path.join(BASE_DIR, `${safe}.json`)
}

async function ensureDir() {
  await fs.mkdir(BASE_DIR, { recursive: true }).catch(() => {})
}

export async function loadConversations(email: string): Promise<Conversation[]> {
  try {
    await ensureDir()
    const file = emailToFile(email)
    const buf = await fs.readFile(file, 'utf-8').catch(() => '')
    if (!buf) return []
    const json = JSON.parse(buf)
    return Array.isArray(json) ? (json as Conversation[]) : []
  } catch {
    return []
  }
}

export async function saveConversations(email: string, list: Conversation[]): Promise<void> {
  await ensureDir()
  const file = emailToFile(email)
  await fs.writeFile(file, JSON.stringify(list, null, 2), 'utf-8')
}

export async function getConversation(email: string, id: string): Promise<Conversation | undefined> {
  const list = await loadConversations(email)
  return list.find(c => c.id === id)
}

export async function upsertConversation(email: string, incoming: Conversation): Promise<Conversation> {
  const list = await loadConversations(email)
  const idx = list.findIndex(c => c.id === incoming.id)
  if (idx === -1) list.push(incoming)
  else list[idx] = incoming
  await saveConversations(email, list)
  return incoming
}

export async function deleteConversation(email: string, id: string): Promise<void> {
  const list = await loadConversations(email)
  const filtered = list.filter(c => c.id !== id)
  await saveConversations(email, filtered)
}
