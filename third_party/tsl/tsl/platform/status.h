/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_TSL_PLATFORM_STATUS_H_
#define TENSORFLOW_TSL_PLATFORM_STATUS_H_

#include <functional>
#include <iosfwd>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/stack_frame.h"
#include "tsl/platform/types.h"
#include "tsl/protobuf/error_codes.pb.h"

// Include appropriate platform-dependent parts of status.
#if defined(PLATFORM_GOOGLE)
#include "tsl/platform/google/status.h"  // IWYU pragma: export
#else
#include "tsl/platform/default/status.h"  // IWYU pragma: export
#endif

namespace tsl {

#if TF_HAS_CPP_ATTRIBUTE(nodiscard)
class [[nodiscard]] Status;
#endif

typedef SourceLocationImpl SourceLocation;

namespace errors {
typedef ::tensorflow::error::Code Code;
}  // namespace errors
namespace error {
typedef ::tensorflow::error::Code Code;
}  // namespace error
}  // namespace tsl

// Transparent comparison between tensorflow::error::Code protobuf enum and
// absl::Status.
//
// The longer term objective is to delete these when we have done the transition
// to absl::Status.
namespace tensorflow::error {
inline bool operator==(const ::tensorflow::error::Code& c1,
                       const absl::StatusCode& c2) {
  return static_cast<int>(c1) == static_cast<int>(c2);
}

inline bool operator!=(const ::tensorflow::error::Code& c1,
                       const absl::StatusCode& c2) {
  return static_cast<int>(c1) != static_cast<int>(c2);
}
}  // namespace tensorflow::error

namespace absl {
inline bool operator==(const ::absl::StatusCode& c1,
                       const ::tensorflow::error::Code& c2) {
  return static_cast<int>(c1) == static_cast<int>(c2);
}

inline bool operator!=(const ::absl::StatusCode& c1,
                       const ::tensorflow::error::Code& c2) {
  return static_cast<int>(c1) != static_cast<int>(c2);
}
}  // namespace absl

namespace tsl {

/// @ingroup core
/// Denotes success or failure of a call in Tensorflow.
class Status {
 public:
  /// Create a success status.
  Status() {}
  ~Status();  // Not inlined to save code space

  /// \brief Create a status with the specified error code and msg as a
  /// human-readable string containing more detailed information.
  Status(absl::StatusCode code, absl::string_view msg,
         SourceLocation loc = SourceLocation::current());
  // Deprecated constructor using the Tensorflow protobuf enum error code.
  Status(tsl::errors::Code code, absl::string_view msg,
         SourceLocation loc = SourceLocation::current());

  /// Copy the specified status.
  Status(const Status& s);
  Status& operator=(const Status& s);
#ifndef SWIG
  Status(Status&& s, SourceLocation loc = SourceLocation::current()) noexcept;
  Status& operator=(Status&& s) noexcept;
#endif  // SWIG

  /// Returns true iff the status indicates success.
  bool ok() const { return (state_ == nullptr); }

  tsl::error::Code code() const {
    return ok() ? tensorflow::error::OK : state_->code;
  }

  const std::string& error_message() const {
    return ok() ? empty_string() : state_->msg;
  }

  bool operator==(const Status& x) const;
  bool operator!=(const Status& x) const;

  /// \brief If `ok()`, stores `new_status` into `*this`.  If `!ok()`,
  /// preserves the current status, but may augment with additional
  /// information about `new_status`.
  ///
  /// Convenient way of keeping track of the first error encountered.
  /// Instead of:
  ///   `if (overall_status.ok()) overall_status = new_status`
  /// Use:
  ///   `overall_status.Update(new_status);`
  void Update(const Status& new_status);

  /// \brief Return a string representation of this status suitable for
  /// printing. Returns the string `"OK"` for success.
  ///
  /// By default, it returns combination of the error code name, the message and
  /// any associated payload messages. This string is designed simply to be
  /// human readable and its exact format should not be load bearing. Do not
  /// depend on the exact format of the result of `ToString()` which is subject
  /// to change.
  std::string ToString() const;

  // Ignores any errors. This method does nothing except potentially suppress
  // complaints from any tools that are checking that errors are not dropped on
  // the floor.
  void IgnoreError() const;

  //----------------------------------------------------------------------------
  // Payload Management APIs (Cloned from absl::Status)
  //----------------------------------------------------------------------------
  // A payload may be attached to a status to provide additional context to an
  // error that may not be satisfied by an existing `tsl::error::Code`.
  // Typically, this payload serves one of several purposes:
  //
  //   * It may provide more fine-grained semantic information about the error
  //     to facilitate actionable remedies.
  //   * It may provide human-readable contexual information that is more
  //     appropriate to display to an end user.
  //
  // A payload consists of a [key,value] pair, where the key is a string
  // referring to a unique "type URL" and the value is an object of type
  // `absl::Cord` to hold the contextual data.
  //
  // The "type URL" should be unique and follow the format of a URL
  // (https://en.wikipedia.org/wiki/URL) and, ideally, provide some
  // documentation or schema on how to interpret its associated data. For
  // example, the default type URL for a protobuf message type is
  // "type.googleapis.com/packagename.messagename". Other custom wire formats
  // should define the format of type URL in a similar practice so as to
  // minimize the chance of conflict between type URLs.
  // Users should ensure that the type URL can be mapped to a concrete
  // C++ type if they want to deserialize the payload and read it effectively.
  //
  // To attach a payload to a status object, call `Status::SetPayload()`,
  // passing it the type URL and an `absl::Cord` of associated data. Similarly,
  // to extract the payload from a status, call `Status::GetPayload()`. You
  // may attach multiple payloads (with differing type URLs) to any given
  // status object, provided that the status is currently exhibiting an error
  // code (i.e. is not OK).
  // TODO(b/197552541): Use absl::Cord for payload value type.

  // The Payload-related APIs are cloned from absl::Status.
  //
  // Returns the payload of a status given its unique `type_url` key, if
  // present.
  absl::optional<absl::Cord> GetPayload(absl::string_view type_url) const;

  // Sets the payload for a non-ok status using a `type_url` key, overwriting
  // any existing payload for that `type_url`.
  //
  // This function does nothing if the Status is ok.
  void SetPayload(absl::string_view type_url, absl::Cord payload);

  // Erases the payload corresponding to the `type_url` key.  Returns `true` if
  // the payload was present.
  bool ErasePayload(absl::string_view type_url);

  // Iterates over the stored payloads and calls the
  // `visitor(type_key, payload)` callable for each one.
  //
  // The order of calls to `visitor()` is not specified and may change at
  // any time and any mutation on the same Status object during visitation is
  // forbidden and could result in undefined behavior.
  void ForEachPayload(
      absl::FunctionRef<void(absl::string_view, const absl::Cord&)> visitor)
      const;

  // Sets the stack frame associated with this status object.
  // Stack traces are only kept and returned via GetStackTrace() if
  // !this->ok().
  void SetStackTrace(std::vector<StackFrame>);

  // Retrieve an associated stack frame for a non-OK status that was
  // set via SetStackTrace().
  std::vector<StackFrame> GetStackTrace() const;

  absl::Span<const SourceLocation> GetSourceLocations() const;

 private:
  friend Status FromAbslStatus(const absl::Status& s, SourceLocation loc);

  void MaybeAddSourceLocation(SourceLocation loc);

  static const std::string& empty_string();
  struct State {
    State() TF_ATTRIBUTE_NOINLINE = default;
    ~State() TF_ATTRIBUTE_NOINLINE = default;
    State(const State&) TF_ATTRIBUTE_NOINLINE = default;
    State& operator=(const State&) TF_ATTRIBUTE_NOINLINE = default;

    tsl::error::Code code;
    std::string msg;
    std::unordered_map<std::string, absl::Cord> payloads;
    absl::InlinedVector<SourceLocation, 4> source_locations;
    std::vector<StackFrame> stack_trace;
  };

  // OK status has a `NULL` state_.  Otherwise, `state_` points to
  // a `State` structure containing the error code and message(s)
  std::unique_ptr<State> state_;

  void SlowCopyFrom(const State* src);
  State* NewStateFromNonOKStatus(const Status& s);
};

// OkStatus()
//
// Returns an OK status, equivalent to a default constructed instance. Prefer
// usage of `OkStatus()` when constructing such an OK status.
Status OkStatus();

Status FromAbslStatus(const absl::Status& s,
                      SourceLocation loc = SourceLocation::current());
absl::Status ToAbslStatus(const ::tsl::Status& s,
                          SourceLocation loc = SourceLocation::current());

// TODO(b/197552541) Move this namespace to errors.h.
namespace errors {

void SetStackTrace(::tsl::Status& status, std::vector<StackFrame> stack_trace);

std::vector<StackFrame> GetStackTrace(const ::tsl::Status& status);
}  // namespace errors

// Helper class to manage multiple child status values.
class StatusGroup {
 public:
  StatusGroup();
  // Constructor to form a StatusGroup from any N set of Status arguments.
  // Usage: StatusGroup({status_a, status_b, status_c});
  StatusGroup(std::initializer_list<Status> statuses);

  // Utility function to mark a Status as derived. By marking derived status,
  // Derived status messages are ignored when reporting errors to end users.
  static Status MakeDerived(const Status& s);
  static bool IsDerived(const Status& s);

  // Enable warning and error log collection for appending to the aggregated
  // status. This function may be called more than once.
  static void ConfigureLogHistory();

  // Returns merged payloads of all statuses. In case multiple statuses have the
  // same payload key, non-derived statuses have priority over derived ones,
  // otherwise one payload value will be chosen in an unspecified but
  // deterministic order.
  // NOTE: The payload marking derived statuses as derived will not be returned.
  std::unordered_map<std::string, absl::Cord> GetPayloads() const;

  // Return a merged status with combined child status messages with a summary.
  Status as_summary_status() const;
  // Return a merged status with combined child status messages with
  // concatenation.
  Status as_concatenated_status() const;

  bool ok() const { return ok_; }

  // Augment this group with the child status `status`.
  void Update(const Status& status);

  // Attach recent warning and error log messages
  void AttachLogMessages();
  bool HasLogMessages() const { return !recent_logs_.empty(); }

 private:
  bool ok_ = true;
  size_t num_ok_ = 0;

  // Maintain a sorted collection of statuses.
  struct CompareStatus {
    bool operator()(const Status& a, const Status& b) const {
      return a.ToString() > b.ToString();
    }
  };
  // Using std::set instead of absl::btree_set to keep size for certain
  // dependent libraries under the limit.
  std::set<Status, CompareStatus> derived_;
  std::set<Status, CompareStatus> non_derived_;

  std::vector<std::string> recent_logs_;  // recent warning and error logs
};

inline Status::Status(const Status& s)
    : state_((s.state_ == nullptr) ? nullptr : NewStateFromNonOKStatus(s)) {}

inline Status& Status::operator=(const Status& s) {
  // The following condition catches both aliasing (when this == &s),
  // and the common case where both s and *this are ok.
  if (state_ != s.state_) {
    SlowCopyFrom(s.state_.get());
  }
  return *this;
}

#ifndef SWIG
inline Status::Status(Status&& s, SourceLocation loc) noexcept
    : state_(std::move(s.state_)) {
  MaybeAddSourceLocation(loc);
}

inline Status& Status::operator=(Status&& s) noexcept {
  if (state_ != s.state_) {
    state_ = std::move(s.state_);
  }
  return *this;
}
#endif  // SWIG

inline bool Status::operator==(const Status& x) const {
  return (this->state_ == x.state_) || (ToString() == x.ToString());
}

inline bool Status::operator!=(const Status& x) const { return !(*this == x); }

/// @ingroup core
std::ostream& operator<<(std::ostream& os, const Status& x);

typedef std::function<void(const Status&)> StatusCallback;

extern tsl::string* TfCheckOpHelperOutOfLine(const ::tsl::Status& v,
                                             const char* msg);

std::string error_name(error::Code code);

inline tsl::string* TfCheckOpHelper(::tsl::Status v, const char* msg) {
  if (v.ok()) return nullptr;
  return TfCheckOpHelperOutOfLine(v, msg);
}

#define TF_DO_CHECK_OK(val, level)                          \
  while (auto* _result = ::tsl::TfCheckOpHelper(val, #val)) \
  LOG(level) << *(_result)

#define TF_CHECK_OK(val) TF_DO_CHECK_OK(val, FATAL)
#define TF_QCHECK_OK(val) TF_DO_CHECK_OK(val, QFATAL)

// DEBUG only version of TF_CHECK_OK.  Compiler still parses 'val' even in opt
// mode.
#ifndef NDEBUG
#define TF_DCHECK_OK(val) TF_CHECK_OK(val)
#else
#define TF_DCHECK_OK(val) \
  while (false && (::tsl::OkStatus() == (val))) LOG(FATAL)
#endif

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_STATUS_H_
