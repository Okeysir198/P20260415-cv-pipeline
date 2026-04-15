import sys
import traceback

passed = 0
failed = 0
errors = []


def run_test(name, func):
    global passed, failed
    try:
        func()
        print(f"  PASS: {name}")
        passed += 1
    except Exception as e:
        traceback.print_exc()
        print(f"  FAIL: {name} — {e}")
        failed += 1
        errors.append((name, str(e)))


def run_all(tests, title=None, header_char="=", exit_on_fail=True):
    global passed, failed, errors
    passed = 0
    failed = 0
    errors = []

    if title:
        print(header_char * 60)
        print(title)
        print(header_char * 60)

    for name, func in tests:
        run_test(name, func)

    total = passed + failed
    if total > 0:
        print(f"\nResults: {passed} passed, {failed} failed")
    else:
        print("\nResults: no tests run")
    if errors:
        print("Failures:")
        for name, err in errors:
            print(f"  - {name}: {err}")

    if exit_on_fail and failed > 0:
        sys.exit(1)
