/**
 * LLM-friendly formatting for memory search and list results.
 *
 * These functions mirror the server's internal formatting logic so that
 * client-side consumers get the same compact representation the server
 * uses when feeding memories into language models.
 *
 * Episodic format matches `string_from_episode_context` on the server.
 * Semantic format matches `_features_to_llm_format` on the server.
 *
 * @packageDocumentation
 */

import type { SemanticMemory, SearchMemoriesResult } from './memmachine-memory.types'

/**
 * Minimal shape required to format an episode.
 *
 * Both search-response episodes ({@link EpisodicMemory}) and list-response
 * episodes ({@link ListEpisodicMemory}) satisfy this shape, so
 * {@link formatEpisodes} works with episodes from either endpoint.
 */
export interface FormattableEpisode {
  created_at?: string | null
  producer_id: string
  content: string
}

const DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'] as const
const MONTHS = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December'
] as const

function _formatDate(date: Date): string {
  const day = DAYS[date.getUTCDay()]
  const month = MONTHS[date.getUTCMonth()]
  const dd = String(date.getUTCDate()).padStart(2, '0')
  const year = date.getUTCFullYear()
  return `${day}, ${month} ${dd}, ${year}`
}

function _formatTime(date: Date): string {
  let hours = date.getUTCHours()
  const minutes = String(date.getUTCMinutes()).padStart(2, '0')
  const ampm = hours >= 12 ? 'PM' : 'AM'
  hours = hours % 12 || 12
  return `${String(hours).padStart(2, '0')}:${minutes} ${ampm}`
}

/**
 * Format episodic memories as an LLM-friendly string.
 *
 * Each episode is rendered as:
 * ```
 * [Monday, January 01, 2024 at 01:30 PM] producer_id: "content"
 * ```
 *
 * This mirrors the server's `string_from_episode_context` output.
 *
 * Accepts any object with `created_at`, `producer_id`, and `content`,
 * so both search-response and list-response episodes work without
 * additional adaptation.
 *
 * @param episodes - Episodic memory entries from a search or list result.
 * @returns Newline-terminated formatted string (empty string when episodes is empty).
 */
export function formatEpisodes(episodes: FormattableEpisode[]): string {
  let result = ''
  for (const episode of episodes) {
    if (episode.created_at) {
      const date = new Date(episode.created_at)
      const dateStr = _formatDate(date)
      const timeStr = _formatTime(date)
      result += `[${dateStr} at ${timeStr}] ${episode.producer_id}: ${JSON.stringify(episode.content)}\n`
    } else {
      result += `${episode.producer_id}: ${JSON.stringify(episode.content)}\n`
    }
  }
  return result
}

/**
 * Format semantic memories as a compact JSON string.
 *
 * Produces a `{tag: {feature_name: value}}` structure, omitting all
 * metadata for context efficiency. This mirrors the server's
 * `_features_to_llm_format` output.
 *
 * @param features - Semantic memory entries from a search or list result.
 * @returns JSON string of the grouped features.
 */
export function formatSemanticMemories(features: SemanticMemory[]): string {
  const structured = Object.create(null) as Record<string, Record<string, string>>
  for (const feature of features) {
    if (!(feature.tag in structured)) {
      structured[feature.tag] = Object.create(null) as Record<string, string>
    }
    structured[feature.tag]![feature.feature_name] = feature.value
  }
  return JSON.stringify(structured)
}

/**
 * Format a search result as an LLM-friendly string.
 *
 * Combines episodic and semantic memories from a {@link SearchMemoriesResult}
 * returned by `MemMachineMemory.search()`.
 *
 * @param result - A `SearchMemoriesResult` object.
 * @returns Formatted string combining both memory types.
 */
export function formatSearchResult(result: SearchMemoriesResult): string {
  const sections: string[] = []

  if (result.content.episodic_memory) {
    const episodes = [
      ...result.content.episodic_memory.long_term_memory.episodes,
      ...result.content.episodic_memory.short_term_memory.episodes
    ]
    if (episodes.length > 0) {
      sections.push(`[Episodic Memory]\n${formatEpisodes(episodes)}`)
    }
  }

  if (result.content.semantic_memory && result.content.semantic_memory.length > 0) {
    sections.push(`[Semantic Memory]\n${formatSemanticMemories(result.content.semantic_memory)}`)
  }

  return sections.join('\n')
}
