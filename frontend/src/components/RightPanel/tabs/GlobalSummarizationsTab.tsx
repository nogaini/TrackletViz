import { useState } from 'react';
import SpatialSubTab from './summarizations/SpatialSubTab';

const SUBTABS = ['Spatial'];

export default function GlobalSummarizationsTab() {
  const [activeSubTab, setActiveSubTab] = useState(0);

  return (
    <div className="flex flex-col h-full">
      <div className="flex border-b border-gray-700 shrink-0 px-2 pt-1 gap-1">
        {SUBTABS.map((label, i) => (
          <button
            key={label}
            onClick={() => setActiveSubTab(i)}
            className={`px-3 py-1.5 text-xs font-medium rounded-t transition-colors ${
              activeSubTab === i
                ? 'bg-gray-700 text-white'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="flex-1 overflow-hidden">
        {activeSubTab === 0 && <SpatialSubTab />}
      </div>
    </div>
  );
}
