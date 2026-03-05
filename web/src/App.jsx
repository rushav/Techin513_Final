import { Sidebar } from './components/ui/Sidebar'
import { HeroSection } from './components/sections/HeroSection'
import { DataGenSection } from './components/sections/DataGenSection'
import { SignalProcessingSection } from './components/sections/SignalProcessingSection'
import { ValidationSection } from './components/sections/ValidationSection'
import { FeaturesSection } from './components/sections/FeaturesSection'
import { MLSection } from './components/sections/MLSection'
import { ResultsSection } from './components/sections/ResultsSection'
import { AblationSection } from './components/sections/AblationSection'
import { ExplorerSection } from './components/sections/ExplorerSection'

export default function App() {
  return (
    <div className="flex min-h-screen">
      <Sidebar />

      <main className="flex-1 px-6 md:px-12 lg:px-16 max-w-5xl mx-auto w-full">
        <HeroSection />
        <div className="h-px bg-gradient-to-r from-transparent via-border to-transparent my-2" />
        <DataGenSection />
        <div className="h-px bg-gradient-to-r from-transparent via-border to-transparent my-2" />
        <SignalProcessingSection />
        <div className="h-px bg-gradient-to-r from-transparent via-border to-transparent my-2" />
        <ValidationSection />
        <div className="h-px bg-gradient-to-r from-transparent via-border to-transparent my-2" />
        <FeaturesSection />
        <div className="h-px bg-gradient-to-r from-transparent via-border to-transparent my-2" />
        <MLSection />
        <div className="h-px bg-gradient-to-r from-transparent via-border to-transparent my-2" />
        <ResultsSection />
        <div className="h-px bg-gradient-to-r from-transparent via-border to-transparent my-2" />
        <AblationSection />
        <div className="h-px bg-gradient-to-r from-transparent via-border to-transparent my-2" />
        <ExplorerSection />

        <footer className="py-16 text-center text-xs text-slate-600">
          <div>TECHIN 513 Final Project · Sleep Environment Signal Processing</div>
          <div className="font-mono mt-1">Generated with seed = 42 · n_sessions = 2500</div>
        </footer>
      </main>
    </div>
  )
}
