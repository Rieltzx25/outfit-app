// Unified 35-class wearable taxonomy.
// Source: OpenImages V7 (Google) + Fashionpedia (CVPR 2020) merged.
// Order MUST match training data.yaml `names` field — model output class index uses this order.
export const OUTFIT_LABELS = [
  // tops 0-7
  'shirt', 't-shirt', 'tank-top', 'sweater', 'cardigan', 'jacket', 'coat', 'vest',
  // one-piece 8-9
  'dress', 'jumpsuit',
  // bottoms 10-13
  'pants', 'jeans', 'shorts', 'skirt',
  // legwear 14-15
  'tights', 'sock',
  // footwear 16-19
  'shoe', 'sandal', 'boot', 'high-heels',
  // headwear 20-22
  'hat', 'helmet', 'hair-accessory',
  // eyewear 23-24
  'glasses', 'sunglasses',
  // neckwear 25-26
  'tie', 'scarf',
  // jewelry & wrist 27-30
  'necklace', 'earrings', 'watch', 'bracelet',
  // other 31-34
  'belt', 'glove', 'bag', 'mask',
] as const

export type OutfitLabel = typeof OUTFIT_LABELS[number]

export const OUTFIT_GROUPS = [
  { name: 'tops',         indices: [0, 1, 2, 3, 4, 7] },
  { name: 'outerwear',    indices: [5, 6] },
  { name: 'bottoms',      indices: [10, 11, 12, 13] },
  { name: 'one-piece',    indices: [8, 9] },
  { name: 'legwear',      indices: [14, 15] },
  { name: 'footwear',     indices: [16, 17, 18, 19] },
  { name: 'headwear',     indices: [20, 21, 22] },
  { name: 'eyewear',      indices: [23, 24] },
  { name: 'neck',         indices: [25, 26] },
  { name: 'jewelry',      indices: [27, 28, 29, 30] },
  { name: 'accessories',  indices: [31, 32, 33, 34] },
] as const

// 35 colors, perceptually distinct, grouped by category for visual coherence.
export const OUTFIT_COLORS = [
  // tops — pinks/red
  '#f43f5e', '#ec4899', '#e11d48', '#be185d', '#db2777', '#fb7185', '#9f1239', '#fda4af',
  // one-piece — purples
  '#a855f7', '#7c3aed',
  // bottoms — blues
  '#3b82f6', '#1d4ed8', '#0ea5e9', '#0284c7',
  // legwear — cyans
  '#06b6d4', '#0891b2',
  // footwear — teals/greens
  '#14b8a6', '#10b981', '#059669', '#22c55e',
  // headwear — yellows
  '#eab308', '#ca8a04', '#facc15',
  // eyewear — oranges
  '#f97316', '#ea580c',
  // neck — limes
  '#84cc16', '#65a30d',
  // jewelry — gold/amber
  '#f59e0b', '#fbbf24', '#d97706', '#fde68a',
  // accessories — slates
  '#64748b', '#475569', '#94a3b8', '#cbd5e1',
]
