#ifndef PARSE_SCHEDULE_H
#define PARSE_SCHEDULE_H

#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

// Parse a comma-separated string of floats into a vector.
// Returns defaults if the string is empty or all tokens fail to parse.
inline std::vector<float> parse_float_schedule(const std::string& str, const std::vector<float>& defaults)
{
    std::vector<float> result;
    if(!str.empty())
    {
        std::stringstream ss(str);
        std::string token;
        while(std::getline(ss, token, ','))
        {
            try { result.push_back(std::stof(token)); }
            catch(std::exception &) {}
        }
    }
    return result.empty() ? defaults : result;
}

// Parse a comma-separated string of ints into a vector.
// Returns defaults if the string is empty or all tokens fail to parse.
inline std::vector<int> parse_int_schedule(const std::string& str, const std::vector<int>& defaults)
{
    std::vector<int> result;
    if(!str.empty())
    {
        std::stringstream ss(str);
        std::string token;
        while(std::getline(ss, token, ','))
        {
            try { result.push_back(std::stoi(token)); }
            catch(std::exception &) {}
        }
    }
    return result.empty() ? defaults : result;
}

// Access a schedule vector by epoch index (0-based).
// If the index exceeds the schedule length, returns the last element.
template<typename T>
inline T schedule_value(const std::vector<T>& schedule, int epoch_index)
{
    if(schedule.empty())
        return T();
    return (size_t)epoch_index < schedule.size() ? schedule[epoch_index] : schedule.back();
}

#endif
