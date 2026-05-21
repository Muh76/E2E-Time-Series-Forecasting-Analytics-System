import { Component, type ErrorInfo, type ReactNode } from "react";
import "./ErrorBoundary.css";

interface Props {
  children: ReactNode;
  /** Optional custom fallback rendered instead of the default error card. */
  fallback?: ReactNode;
  /** Label shown in the error card header (default: "Something went wrong"). */
  label?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/**
 * Class-based error boundary that catches render errors anywhere in its
 * subtree, logs them, and displays a recoverable fallback UI instead of
 * crashing the entire SPA.
 *
 * Usage:
 *   <ErrorBoundary>
 *     <SomeComponent />
 *   </ErrorBoundary>
 */
export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("[ErrorBoundary] Uncaught render error:", error, info.componentStack);
  }

  private handleReset = (): void => {
    this.setState({ hasError: false, error: null });
  };

  render(): ReactNode {
    const { hasError, error } = this.state;
    const { children, fallback, label = "Something went wrong" } = this.props;

    if (!hasError) return children;
    if (fallback) return fallback;

    return (
      <div className="error-boundary">
        <div className="error-boundary__card">
          <h2 className="error-boundary__title">{label}</h2>
          <p className="error-boundary__message">
            {error?.message ?? "An unexpected error occurred. The page could not be rendered."}
          </p>
          <button className="error-boundary__reset" onClick={this.handleReset}>
            Try again
          </button>
        </div>
      </div>
    );
  }
}
