interface Props {
  onClick: () => void;
  disabled?: boolean;
}

export default function VerifyAction({ onClick, disabled }: Props) {
  return (
    <button
      type="button"
      onClick={onClick}
      onKeyDown={(event) => event.stopPropagation()}
      disabled={disabled}
      className="mt-1 rounded border border-teal-700 px-2 py-1 text-xs font-semibold text-teal-800 dark:text-teal-300 disabled:opacity-40"
    >
      Verify with Agent
    </button>
  );
}
