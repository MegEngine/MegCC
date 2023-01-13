/**
 * \file compiler/include/compiler/Common/TContext.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "utils.h"

namespace megcc {
struct CCOperand {
    std::vector<size_t> shape;
    std::string dtype;
    float scale = -1.f;
    size_t nr_elem() {
        if (shape.size() == 0) {
            return 0;
        }
        size_t res = 1;
        for (auto s : shape) {
            res *= s;
        }
        return res;
    }

    std::string to_string() {
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            ss << shape[i] << ", ";
        }
        ss << dtype;
        if (scale >= 0) {
            ss << ":" << scale;
        }
        ss << "]";
        return ss.str();
    }

    std::string name_string() {
        std::stringstream ss;
        for (size_t i = 0; i < shape.size(); ++i) {
            ss << shape[i] << "_";
        }
        ss << dtype;
        return ss.str();
    }
};
class CCAttr {
    enum Type { STRING = 0, INT = 1, FLOAT = 2, OPERAND = 3, BOOL = 4 };

public:
    CCAttr() = default;

    CCAttr(const char* value) {
        //! TODO: new unique_ptr may be not efficient
        std::string value_str = std::string(value);
        mHolder = std::make_unique<AnyHolder<std::string>>(value_str);
        mType = Type::STRING;
        mLength = value_str.size();
    }

    CCAttr(std::string value) {
        //! TODO: new unique_ptr may be not efficient
        mHolder = std::make_unique<AnyHolder<std::string>>(value);
        mType = Type::STRING;
        mLength = value.size();
    }

    CCAttr(bool value) {
        //! TODO: new unique_ptr may be not efficient
        mHolder = std::make_unique<AnyHolder<bool>>(value);
        mType = Type::BOOL;
        mLength = sizeof(bool);
    }

    CCAttr(const CCOperand& value) {
        //! TODO: new unique_ptr may be not efficient
        mHolder = std::make_unique<AnyHolder<CCOperand>>(value);
        mType = Type::OPERAND;
        mLength = sizeof(CCOperand);
    }

    template <typename T,
              std::enable_if_t<std::is_integral<T>::value, bool> = true>
    CCAttr(T value) {
        //! TODO: new unique_ptr may be not efficient
        mHolder = std::make_unique<AnyHolder<T>>(value);
        mType = Type::INT;
        mLength = sizeof(T);
    }

    template <typename T,
              std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
    CCAttr(T value) {
        //! TODO: new unique_ptr may be not efficient
        mHolder = std::make_unique<AnyHolder<T>>(value);
        mType = Type::FLOAT;
        mLength = sizeof(T);
    }

    CCAttr(const CCAttr& attr) {
        mHolder = attr.mHolder->clone();
        mType = attr.mType;
        mLength = attr.mLength;
    }

    CCAttr& operator=(const CCAttr& attr) {
        mHolder = attr.mHolder->clone();
        mType = attr.mType;
        mLength = attr.mLength;
        return *this;
    }

    class HolderBase {
    public:
        virtual ~HolderBase() = default;
        virtual std::unique_ptr<HolderBase> clone() = 0;
    };

    template <class T>
    class AnyHolder : public HolderBase {
    public:
        AnyHolder(const T value) : mValue(value) {}
        virtual std::unique_ptr<HolderBase> clone() override {
            return std::make_unique<AnyHolder>(mValue);
        }

    public:
        T mValue;
    };

    template <class T,
              std::enable_if_t<std::is_integral<T>::value, bool> = true>
    T AsInt() const {
        if (mType != Type::INT || sizeof(T) != mLength) {
            fprintf(stderr,
                    "get CCAttr with wrong type mType %d != %d || size %d != "
                    "%d\n",
                    (int)mType, (int)Type::INT, (int)sizeof(T), (int)mLength);
            abort();
        }
        return static_cast<CCAttr::AnyHolder<T>*>(mHolder.get())->mValue;
    }

    template <class T,
              std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
    T AsFloat() const {
        if (mType != Type::FLOAT || sizeof(T) != mLength) {
            fprintf(stderr, "get CCAttr with wrong type \n");
            abort();
        }
        return static_cast<CCAttr::AnyHolder<T>*>(mHolder.get())->mValue;
    }

    std::string AsString() const {
        if (mType != Type::STRING) {
            fprintf(stderr, "get CCAttr with wrong type \n");
            abort();
        }
        return static_cast<CCAttr::AnyHolder<std::string>*>(mHolder.get())
                ->mValue;
    }

    CCOperand AsOperand() const {
        if (mType != Type::OPERAND) {
            fprintf(stderr, "get CCAttr with wrong type \n");
            abort();
        }
        return static_cast<CCAttr::AnyHolder<CCOperand>*>(mHolder.get())
                ->mValue;
    }

    bool AsBool() const {
        if (mType != Type::BOOL) {
            fprintf(stderr, "get CCAttr with wrong type \n");
            abort();
        }
        return static_cast<CCAttr::AnyHolder<bool>*>(mHolder.get())->mValue;
    }

private:
    Type mType;
    uint32_t mLength;
    std::unique_ptr<HolderBase> mHolder;
};

struct TContext {
    virtual int8_t getAttrInt8(const std::string& attrName) = 0;
    virtual uint8_t getAttrUInt8(const std::string& attrName) = 0;
    virtual int32_t getAttrInt(const std::string& attrName) = 0;
    virtual uint32_t getAttrUInt(const std::string& attrName) = 0;
    virtual uint64_t getAttrUInt64(const std::string& attrName) = 0;
    virtual int64_t getAttrInt64(const std::string& attrName) = 0;
    virtual float getAttrFloat(const std::string& attrName) = 0;
    virtual double getAttrDouble(const std::string& attrName) = 0;
    virtual std::string getAttrStr(const std::string& attrName) = 0;
    virtual CCOperand getAttrOprand(const std::string& attrName) = 0;
    virtual bool getAttrBool(const std::string& attrName) = 0;
    virtual bool haveAttr(const std::string& attrName) = 0;
    virtual void setAttr(const std::string& attrName, const CCAttr& attr) = 0;
    virtual ~TContext() = default;
};

#define DefineFunction(type_, ctype_, As_)                        \
    ctype_ getAttr##type_(const std::string& attrName) override { \
        return getAttr(attrName).As_<ctype_>();                   \
    }

struct CodeGenContext final : public TContext {
    DefineFunction(Int8, int8_t, AsInt);
    DefineFunction(UInt8, uint8_t, AsInt);
    DefineFunction(Int, int32_t, AsInt);
    DefineFunction(UInt, uint32_t, AsInt);
    DefineFunction(Int64, int64_t, AsInt);
    DefineFunction(UInt64, uint64_t, AsInt);
    DefineFunction(Float, float, AsFloat);
    DefineFunction(Double, double, AsFloat);

    std::string getAttrStr(const std::string& attrName) override {
        return getAttr(attrName).AsString();
    }
    CCOperand getAttrOprand(const std::string& attrName) override {
        return getAttr(attrName).AsOperand();
    }
    bool getAttrBool(const std::string& attrName) override {
        return getAttr(attrName).AsBool();
    }

    bool haveAttr(const std::string& attrName) override {
        return attrMap.find(attrName) != attrMap.end();
    }

    void setAttr(const std::string& name, const CCAttr& attr) override {
        attrMap[name] = attr;
    }

    CCAttr getAttr(const std::string& name) {
        if (haveAttr(name)) {
            return attrMap[name];
        } else {
            fprintf(stderr, "not find CCAttr %s\n", name.c_str());
            abort();
        }
    }

    CodeGenContext(const std::unordered_map<std::string, CCAttr>& attr_map)
            : attrMap(attr_map) {}

    CodeGenContext() = default;

private:
    std::unordered_map<std::string, CCAttr> attrMap;
};

#undef DefineFunction
}  // namespace megcc
   // vim: syntax=cpp.doxygen
